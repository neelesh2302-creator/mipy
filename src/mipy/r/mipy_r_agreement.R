# ======================================================================
#  MIPY R agreement: runs the original R implementations on the masked
#  matrices that MIPY wrote, so the Python versions can be compared.
#  Usage: Rscript mipy_r_agreement.R <folder>
#  Reads  <folder>/config.csv, X_<mech>.csv, Z_<mech>.csv,
#         lambda_<mech>.txt, pred_<mech>.csv
#  Writes si_<mech>.csv, mf_<mech>.csv, mice_<mech>_<l>.csv (mice's norm.boot;
#         if it stops, see the note before the loop),
#         minprob_<mech>.csv, qrilc_<mech>.csv, par_minprob_<mech>.csv,
#         par_qrilc_<mech>.csv, status.csv, versions.csv
# ======================================================================
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "."
cfg <- read.csv(file.path(dir, "config.csv"), stringsAsFactors = FALSE)
get <- function(k) cfg$value[cfg$key == k]
mechs <- strsplit(get("mechs"), " ")[[1]]
seed <- as.integer(get("seed")); m <- as.integer(get("m"))
mice_maxit <- as.integer(get("mice_maxit"))
mf_ntree <- as.integer(get("mf_ntree")); mf_maxiter <- as.integer(get("mf_maxiter"))
do <- strsplit(get("run"), " ")[[1]]
repos <- "https://cloud.r-project.org"
status <- data.frame(step = character(), mech = character(), status = character(),
                     seconds = numeric(), stringsAsFactors = FALSE)
note <- function(step, mech, st, sec = NA) {
  status[nrow(status) + 1, ] <<- list(step, mech, st, sec)
  write.csv(status, file.path(dir, "status.csv"), row.names = FALSE)
}

ensure <- function(p) {
  if (requireNamespace(p, quietly = TRUE)) return(TRUE)
  ok <- tryCatch({
    if (p == "imputeLCMD") {
      for (b in c("pcaMethods", "impute")) if (!requireNamespace(b, quietly = TRUE)) {
        if (!requireNamespace("BiocManager", quietly = TRUE))
          install.packages("BiocManager", repos = repos, quiet = TRUE)
        BiocManager::install(b, update = FALSE, ask = FALSE, quiet = TRUE)
      }
    }
    install.packages(p, repos = repos, quiet = TRUE)
    requireNamespace(p, quietly = TRUE)
  }, error = function(e) FALSE)
  isTRUE(ok)
}
pk <- c(softImpute = "softImpute", missForest = "missForest", mice = "mice", lcmd = "imputeLCMD")
have <- sapply(pk, ensure)
for (k in names(pk)) note(paste0("package:", pk[[k]]), "", if (have[[k]]) "ok" else "unavailable")
vers <- data.frame(package = c("R", pk),
                   version = c(paste(R.version$major, R.version$minor, sep = "."),
                               sapply(pk, function(p) tryCatch(as.character(packageVersion(p)),
                                                              error = function(e) NA))))
write.csv(vers, file.path(dir, "versions.csv"), row.names = FALSE)

rd <- function(f) as.matrix(read.csv(file.path(dir, f), check.names = FALSE))
wr <- function(x, f) write.csv(x, file.path(dir, f), row.names = FALSE, na = "")
timed <- function(step, mech, expr) {
  t0 <- proc.time()[["elapsed"]]
  r <- tryCatch({ force(expr); "ok" }, error = function(e) paste("error:", conditionMessage(e)))
  note(step, mech, r, proc.time()[["elapsed"]] - t0)
  invisible(r)
}

# mice's norm.boot fits ordinary least squares (QR) on a bootstrap sample.
# When a protein has few observed values, a bootstrap sample can have fewer
# distinct rows than predictors; lm.fit then returns NA coefficients, the
# imputations become NA and mice stops. In that case the step is rerun with
# mice's own ridge option for the same fit (ls.meth = "ridge", penalty
# 1e-5 x diag(X'X), see ?mice::estimice), recorded as "mice-ridge". If a
# variable has at most (predictors + 1) observed values, norm.boot's residual
# SD is undefined even then, and mice's Bayesian "norm" method (which sets
# aliased coefficients to 0) is used instead, recorded as "mice-norm".

for (mech in mechs) {
  X <- rd(paste0("X_", mech, ".csv"))
  if ("softImpute" %in% do && have[["softImpute"]]) timed("softImpute", mech, {
    Z <- rd(paste0("Z_", mech, ".csv"))
    lam <- as.numeric(readLines(file.path(dir, paste0("lambda_", mech, ".txt"))))
    fit <- softImpute::softImpute(Z, rank.max = min(dim(Z)) - 1, lambda = lam,
                                  type = "svd", thresh = 1e-5, maxit = 200)
    wr(softImpute::complete(Z, fit), paste0("si_", mech, ".csv"))
  })
  if ("missForest" %in% do && have[["missForest"]]) timed("missForest", mech, {
    set.seed(seed)
    mf <- missForest::missForest(as.data.frame(X), ntree = mf_ntree, maxiter = mf_maxiter,
                                 verbose = FALSE)$ximp
    wr(as.matrix(mf), paste0("mf_", mech, ".csv"))
  })
  if ("mice" %in% do && have[["mice"]]) {
    pm <- rd(paste0("pred_", mech, ".csv"))
    dimnames(pm) <- list(colnames(X), colnames(X))
    run_mice <- function(meth, ls) {
      imp <- mice::mice(as.data.frame(X), m = m, method = meth, predictorMatrix = pm,
                        maxit = mice_maxit, seed = seed, printFlag = FALSE, ls.meth = ls)
      for (l in seq_len(m)) wr(as.matrix(mice::complete(imp, l)), paste0("mice_", mech, "_", l, ".csv"))
    }
    for (l in seq_len(m)) unlink(file.path(dir, paste0("mice_", mech, "_", l, ".csv")))
    if (!identical(timed("mice", mech, run_mice("norm.boot", "qr")), "ok"))
      if (!identical(timed("mice-ridge", mech, run_mice("norm.boot", "ridge")), "ok"))
        timed("mice-norm", mech, run_mice("norm", "qr"))
  }
  if ("lcmd" %in% do && have[["lcmd"]]) timed("imputeLCMD", mech, {
    D <- t(X)                                   # features x samples, as imputeLCMD expects
    min.samples <- apply(D, 2, quantile, prob = 0.01, na.rm = TRUE)
    cnt <- apply(!is.na(D), 1, sum) / ncol(D)
    sd.temp <- median(apply(D[which(cnt > 0.5), ], 1, sd, na.rm = TRUE), na.rm = TRUE)
    wr(data.frame(centre = min.samples, sd = sd.temp), paste0("par_minprob_", mech, ".csv"))
    qp <- t(sapply(seq_len(ncol(D)), function(i) {
      x <- D[, i]; pNAs <- length(which(is.na(x))) / length(x); upper.q <- 0.99
      qn <- qnorm(seq((pNAs + 0.001), (upper.q + 0.001), (upper.q - pNAs) / (upper.q * 100)))
      qx <- quantile(x, probs = seq(0.001, (upper.q + 0.001), 0.01), na.rm = TRUE)
      cf <- coef(lm(qx ~ qn))
      c(mean = cf[[1]], sd = cf[[2]], upper = qnorm(pNAs + 0.001, cf[[1]], cf[[2]]))
    }))
    wr(qp, paste0("par_qrilc_", mech, ".csv"))
    set.seed(seed)
    mp <- suppressMessages(capture.output(r1 <- imputeLCMD::impute.MinProb(D, q = 0.01, tune.sigma = 1)))
    wr(t(r1), paste0("minprob_", mech, ".csv"))
    set.seed(seed)
    r2 <- imputeLCMD::impute.QRILC(D, tune.sigma = 1)[[1]]
    wr(t(r2), paste0("qrilc_", mech, ".csv"))
  })
}
note("done", "", "ok")

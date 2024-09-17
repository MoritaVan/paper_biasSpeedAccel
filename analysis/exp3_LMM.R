library(nlme)
library(lme4)
library(MASS)
library(stargazer)
library(broom)
library(plyr)
library(buildmer)
library(emmeans)
library(BayesFactor)
library("bayestestR")
library(stats)


formatRanef <- function(r1) { # receives the ranef structure
  
  sd_r1 <- list()
  
  coln <- colnames(r1)
  
  listout <- list(
    c('Random Effects', ''),
    c("Groups", NROW(r1))
  )  
  
  for (c in coln) {
    if (c %in% colnames(r1)){
      sd_r1[c] <- sd(unlist(r1[c])) 
    } else {
      sd_r1[c] <- NaN
    }
    listout <- append(listout, list(c(sprintf("sd(%s)",c), sprintf("%.2f",sd_r1[c]))))  
    
  }
  listout
}

xVar <- c('sub', 'condition', 'trial', 'exp', 'sub_txt', 'aSPv_x')
yVar <- c('sub', 'condition', 'trial', 'exp', 'sub_txt', 'aSPv_y')

setwd("~/Experiments/data/outputs/exp3/")

data <- read.csv('exp3_params.csv', sep=',')
data <- data[data$trial>10,]

data$exp <- replicate(nrow(data), 'constantTime')

xAxis <- data[xVar]
yAxis <- data[yVar]

xAxis$axis <- replicate(nrow(xAxis), 'horiz.')
yAxis$axis <- replicate(nrow(xAxis), 'vert.')

new_colnames <- c('sub','condition','trial','exp','sub_txt','aSPv', 'axis')
colnames(xAxis) <- new_colnames
colnames(yAxis) <- new_colnames

dataAll <- rbind(xAxis,yAxis)



v0 <- dataAll$condition
v0[v0=="V1c"] <- 11/sqrt(2)
v0[v0=="V2c"] <- 22/sqrt(2)
v0[v0=="V3c"] <- 33/sqrt(2)
v0[v0=="V1a"] <- 11/sqrt(2)
v0[v0=="V2a"] <- 22/sqrt(2)
v0[v0=="V3a"] <- 33/sqrt(2)
v0[v0=="V1d"] <- 11/sqrt(2)
v0[v0=="V2d"] <- 22/sqrt(2)
v0[v0=="V3d"] <- 33/sqrt(2)
v0 <- as.numeric(v0)
dataAll$v0 <- v0

ac <- dataAll$condition
ac[ac=="V1c"] <- 0/sqrt(2)
ac[ac=="V2c"] <- 0/sqrt(2)
ac[ac=="V3c"] <- 0/sqrt(2)
ac[ac=="V1a"] <- 22/sqrt(2)
ac[ac=="V2a"] <- 22/sqrt(2)
ac[ac=="V3a"] <- 22/sqrt(2)
ac[ac=="V1d"] <- -22/sqrt(2)
ac[ac=="V2d"] <- -22/sqrt(2)
ac[ac=="V3d"] <- -22/sqrt(2)
ac <- as.numeric(ac)
dataAll$accel <- ac


###############################################

#######################
# categorical model

form <- aSPv ~ 1 + condition + (1 + condition|sub)

model <- buildmer(form, data=dataAll,
                  buildmerControl=buildmerControl(direction=c('order','backward'),
                                                  args=list(control=lmerControl(optimizer='bobyqa'))))
formula(model)
aSPv_lmm <- lme(aSPv ~ 1 + condition ,
                random = list(sub = ~ 1),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)

summary(aSPv_lmm)
contrast(emmeans(aSPv_lmm, specs="condition"), "pairwise", adjust='BH') # Benjamini & Hochberg, FDR


# parametric model
form <- aSPv ~ 1 + v0*accel + (1 + v0 + accel|sub)
model <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                       args=list(control=lmerControl(optimizer='bobyqa'))), data=dataAll)
formula(model)
aSPv_lmm <- lme(aSPv ~ 1 + v0 + accel + v0:accel,
                random = list(sub = ~ 1 + v0 ),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm)


# bayesian comparisons between v0 only, acc only, and full models
aSPv_lmm_v0 <- lme(aSPv ~ 1 + v0,
                random = list(sub = ~ 1 + v0 ),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_v0)
aSPv_lmm_accel <- lme(aSPv ~ 1 + accel,
                random = list(sub = ~ 1 + accel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_accel)

aSPv_lmm_full <- lme(aSPv ~ 1 + v0*accel,
                      random = list(sub = ~ 1 + v0 + accel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                      data=dataAll)
summary(aSPv_lmm_full)


bicV0   <- BIC(aSPv_lmm_v0)
bicAc   <- BIC(aSPv_lmm_accel)
bicFull <- BIC(aSPv_lmm_full)

bic_to_bf(c(bicV0, bicAc, bicFull), denominator = bicV0)

# exporting
randomeffects <- data.frame(
  ranef(aSPv_lmm)
)

colnames(randomeffects) <- c('Intercept', 'v0')
v1 <- unique(dataAll$sub)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPv")
fixedeffectsAnti <- data.frame(
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns


write.csv(randomeffects, 'LMM/exp3_cond100Pred_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp3_cond100Pred_lmm_fixedeffectsAnti.csv')



rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "v0")

starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp3_cond100Pred_lmmResults_antiParams.html', 
                          title='Exp 3: Anticipatory Parameters – Parametric Variables',
                          single.row=TRUE,
                          report = "vc*stp",
                          star.cutoffs = c(.01, .001, .0001),
                          ci=TRUE, ci.level=0.95, digits=3,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rAV),
                          dep.var.labels = c("aSPv"),
                          covariate.labels = c(
                            "V0",
                            "Accel", 
                            "V0:Accel",
                            'Constant'))


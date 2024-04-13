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

formatRanef <- function(r1, r2) { # receives the ranef structure
  sd_r1 <- list()
  sd_r2 <- list()
  coln <- append(colnames(r1),colnames(r2))
  coln <- unique(coln)
  
  listout <- list(
    c('Random Effects', '', ''),
    c("Groups", NROW(r1), NROW(r2))
  )
  for (c in coln) {
    if (c %in% colnames(r1)){
      sd_r1[c] <- sd(unlist(r1[c])) 
    } else {
      sd_r1[c] <- NaN
    }
    if (c %in% colnames(r2)){
      sd_r2[c] <- sd(unlist(r2[c]))  
    } else {
      sd_r2[c] <- NaN
    }
    listout <- append(listout, list(c(sprintf("sd(%s)",c), sprintf("%.2f",sd_r1[c]), sprintf("%.2f",sd_r2[c]))))
  }
  listout
}
xVar <- c('sub', 'condition', 'trial', 'exp', 'sub_txt', 'aSPv_x', 'aSPon_x', 'SPlat_x', 'SPacc_x')
yVar <- c('sub', 'condition', 'trial', 'exp', 'sub_txt', 'aSPv_y', 'aSPon_y', 'SPlat_y', 'SPacc_y')

setwd("~/Experiments/data/outputs/exp2ctrl/")

dataCtrl <- read.csv('exp2ctrl_params_cond100Pred.csv', sep=',')
dataCtrl <- dataCtrl[dataCtrl$trial>10,]

dataCtrl$exp <- replicate(nrow(dataCtrl), 'constantTime')


setwd("~/Experiments/data/outputs/exp2/")

data <- read.csv('exp2_params_cond100Pred.csv', sep=',')
data <- data[data$trial>10,]

data$exp <- replicate(nrow(data), 'constantDistance')

maxSub <- max(data$sub)

dataCtrl$sub <- dataCtrl$sub + maxSub
dataCtrl[dataCtrl$sub==maxSub+1,'sub'] <- 1
dataCtrl[dataCtrl$sub==maxSub+2,'sub'] <- 13

xAxisCtrl <- dataCtrl[xVar]
yAxisCtrl <- dataCtrl[yVar]

xAxisCtrl$axis <- replicate(nrow(xAxisCtrl), 'horiz.')
yAxisCtrl$axis <- replicate(nrow(xAxisCtrl), 'vert.')

xAxis <- data[xVar]
yAxis <- data[yVar]

xAxis$axis <- replicate(nrow(xAxis), 'horiz.')
yAxis$axis <- replicate(nrow(xAxis), 'vert.')

new_colnames <- c('sub','condition','trial','exp','sub_txt','aSPv','aSPon','SPlat','SPacc','axis')
colnames(xAxis) <- new_colnames
colnames(yAxis) <- new_colnames
colnames(xAxisCtrl) <- new_colnames
colnames(yAxisCtrl) <- new_colnames

dataAll <- rbind(xAxis,yAxis,xAxisCtrl,yAxisCtrl)

cond <- dataAll$condition
cond[cond=='V1-100_V0-0'] <- 'v11'
cond[cond=='V2-100_V0-0'] <- 'v22'
cond[cond=='V3-100_V0-0'] <- 'v33'
cond[cond=='Va-100_V0-0'] <- 'vacc'
cond[cond=='Vd-100_V0-0'] <- 'vdec'
dataAll$condition <- cond

v0 <- cond
v0[v0=="v11"] <- 11/sqrt(2)
v0[v0=="v22"] <- 22/sqrt(2)
v0[v0=="v33"] <- 33/sqrt(2)
v0[v0=="vacc"] <- 11/sqrt(2)
v0[v0=="vdec"] <- 33/sqrt(2)
v0 <- as.numeric(v0)
dataAll$v0 <- v0

ac <- cond
ac[ac=="v11"] <- 0/sqrt(2)
ac[ac=="v22"] <- 0/sqrt(2)
ac[ac=="v33"] <- 0/sqrt(2)
ac[ac=="vacc"] <- 22/sqrt(2)
ac[ac=="vdec"] <- -22/sqrt(2)
ac <- as.numeric(ac)
dataAll$accel <- ac

## BF analysis
meanAntiVel_v1 <- vector()
meanAntiVel_va <- vector()
meanAntiVel_v3 <- vector()
meanAntiVel_vd <- vector()
meanAntiVel_v2 <- vector()
for (s in unique(dataAll$sub)) {
  meanAntiVel_v1 <- append(meanAntiVel_v1, mean(dataAll$aSPv[(dataAll$sub==s)&(dataAll$condition=="v11")]))
  meanAntiVel_va <- append(meanAntiVel_va, mean(dataAll$aSPv[(dataAll$sub==s)&(dataAll$condition=="vacc")]))
  
  meanAntiVel_v3 <- append(meanAntiVel_v3, mean(dataAll$aSPv[(dataAll$sub==s)&(dataAll$condition=="v33")]))
  meanAntiVel_vd <- append(meanAntiVel_vd, mean(dataAll$aSPv[(dataAll$sub==s)&(dataAll$condition=="vdec")]))
  
  meanAntiVel_v2 <- append(meanAntiVel_v2, mean(dataAll$aSPv[(dataAll$sub==s)&(dataAll$condition=="v22")]))
}

bf_v1va = ttestBF(x = meanAntiVel_v1,meanAntiVel_va, paired=TRUE)
bf_v1va

bf_v2va = ttestBF(x = meanAntiVel_v2,meanAntiVel_va, paired=TRUE)
bf_v2va

bf_v3vd = ttestBF(x = meanAntiVel_v3,meanAntiVel_vd, paired=TRUE)
bf_v3vd

bf_v2vd = ttestBF(x = meanAntiVel_v2,meanAntiVel_vd, paired=TRUE)
bf_v2vd

# testing effect of acceleration vs initial speed only
aSPv_lmm_intercept <- lme(aSPv ~ 1 + exp,
                   random = list(sub = ~ 1 + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                   data=dataAll)
summary(aSPv_lmm_intercept)
aSPv_lmm_v0 <- lme(aSPv ~ 1 + v0*exp,
                random = list(sub = ~ 1 + v0 + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_v0)
aSPv_lmm_accel <- lme(aSPv ~ 1 + accel*exp,
                random = list(sub = ~ 1 + accel + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_accel)
aSPv_lmm <- lme(aSPv ~ 1 + v0*accel*exp,
                random = list(sub = ~ 1 + v0 + accel + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))

bicbase   <- BIC(aSPv_lmm_intercept)
bicV0   <- BIC(aSPv_lmm_v0)
bicAc   <- BIC(aSPv_lmm_accel)
bicFull <- BIC(aSPv_lmm)

bic_to_bf(c(bicbase, bicV0, bicAc, bicFull), denominator = bicbase)
bic_to_bf(c(bicV0, bicAc, bicFull), denominator = bicV0)


aSPon_lmm <- lme(aSPon ~ 1 + v0*accel*exp,
                random = list(sub = ~ 1 + v0 + accel + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPon_lmm)
qqnorm(aSPon_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPon_lmm))

SPlat_lmm <- lme(SPlat ~ 1 + v0*accel*exp,
                 random = list(sub = ~ 1 + v0 + accel + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=dataAll)
summary(SPlat_lmm)
qqnorm(SPlat_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPlat_lmm))

SPacc_lmm <- lme(SPacc ~ 1 + v0*accel*exp,
                 random = list(sub = ~ 1 + v0 + accel + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=dataAll)
summary(SPacc_lmm)
qqnorm(SPacc_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPacc_lmm))




randomeffects <- data.frame(
  rbind.fill(ranef(aSPon_lmm),
             ranef(aSPv_lmm),
             ranef(SPlat_lmm),
             ranef(SPacc_lmm)
  ))

colnames(randomeffects) <- c('Intercept', 'v0', 'accel', 'experiment')
v1 <- unique(dataAll$sub)
randomeffects$sub <- c(v1,v1,v1,v1)
randomeffects$var <- c(rep("aSPon", length(v1)), 
                       rep("aSPv", length(v1)), 
                       rep("SPlat", length(v1)), 
                       rep("SPacc", length(v1)))

columns = c("aSPon","aSPv")
fixedeffectsAnti <- data.frame(
  c1 <- fixef(aSPon_lmm),
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns

columns = c("SPlat","SPacc")
fixedeffectsVGP <- data.frame(
  c1 <- fixef(SPlat_lmm),
  c2 <- fixef(SPacc_lmm)
)
colnames(fixedeffectsVGP) <- columns

write.csv(randomeffects, 'LMM/exp2_cond100Pred_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp2_cond100Pred_lmm_fixedeffectsAnti.csv')
write.csv(fixedeffectsVGP, 'LMM/exp2_cond100Pred_lmm_fixedeffectsVGP.csv')




rSA <- ranef(aSPon_lmm)
colnames(rSA) <- c("Constant", "v0", "accel", "exp[const.Time]")
rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "v0", "accel", "exp[const.Time]")
rLA <- ranef(SPlat_lmm)
colnames(rLA) <- c("Constant", "v0", "accel", "exp[const.Time]")
rPA <- ranef(SPacc_lmm)
colnames(rPA) <- c("Constant", "v0", "accel", "exp[const.Time]")


starAnti.out <- stargazer(aSPon_lmm,aSPv_lmm,
                          out='LMM/exp2_cond100Pred_lmmResults_antiParams.html', 
                          title='Anticipatory Parameters',
                          single.row=FALSE,
                          report = "vc*stp",
                          star.cutoffs = c(.01, .001, .0001),
                          ci=TRUE, ci.level=0.95, digits=3,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rSA,rAV),
                          dep.var.labels = c("aSPon", "aSPv"),
                          covariate.labels = c(
                            "V0",
                            "Accel", 
                            "Exp.[const.Time]",
                            "V0:Accel",
                            "V0:Exp.[const.Time]",
                            "Accel:Exp.[const.Time]",
                            "V0:Accel:Exp.[const.Time]",
                            'Constant'))


starVGP.out <- stargazer(SPlat_lmm,SPacc_lmm,
                         out='LMM/exp2_cond100Pred_lmmResults_VGPparams.html', 
                         title='Visually Guided Parameters',
                         single.row=FALSE,
                         report = "vc*stp",
                         star.cutoffs = c(.01, .001, .0001),
                         ci=TRUE, ci.level=0.95, digits=3,
                         model.numbers = FALSE,
                         omit.stat=c("LL","ser","f", 'aic', 'bic'),
                         keep.stat = c("rsq","f"),
                         add.lines = formatRanef(rLA,rPA),
                         dep.var.labels = c("SPlat", "SPacc"),
                         covariate.labels = c(
                           "V0",
                           "Accel", 
                           "Exp.[const.Time]",
                           "V0:Accel",
                           "V0:Exp.[const.Time]",
                           "Accel:Exp.[const.Time]",
                           "V0:Accel:Exp.[const.Time]",
                           'Constant'))


###############################################
aSPv_lmm <- lme(aSPv ~ 1 + condition*axis,
                random = list(sub = ~ 1 + condition+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm)
contrast(emmeans(aSPv_lmm, specs="condition"), "pairwise", adjust='none')
contrast(emmeans(aSPv_lmm, specs="condition"), "pairwise", adjust='BH') # Benjamini & Hochberg, FDR


aSPon_lmm <- lme(aSPon ~ 1 + condition*exp,
                random = list(sub = ~ 1 + condition + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPon_lmm)

SPlat_lmm <- lme(SPlat ~ 1 + condition*exp,
                 random = list(sub = ~ 1 + condition + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=dataAll)
summary(SPlat_lmm)

SPacc_lmm <- lme(SPacc ~ 1 + condition*exp,
                 random = list(sub = ~ 1 + condition + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=dataAll)
summary(SPacc_lmm)

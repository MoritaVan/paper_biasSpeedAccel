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


###############################################
form <- aSPv ~ 1 + condition*axis*exp + (1 + condition+axis+exp|sub)
# model <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
#                                                        args=list(control=lmerControl(optimizer='bobyqa'))), data=dataAll)
# formula(model)

aSPv_lmm <- lme(aSPv ~ 1 + condition + axis + exp + axis:exp + condition:axis,
                random = list(sub = ~ 1 + condition + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm)
# contrast(emmeans(aSPv_lmm, specs="condition"), "pairwise", adjust='bonferroni')
contrast(emmeans(aSPv_lmm, specs="condition"), "pairwise", adjust='BH') # Benjamini & Hochberg, FDR

rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "v22", "v33", "vacc", "vdec", 'Axis[vert.]')


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp2_cond100Pred_lmmResults_categorical_antiParams.html', 
                          title='Exp 2A-B: Anticipatory Parameters – Categorical Variables',
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
                            "v22","v33","vacc","vdec",
                            "Axis[vert.]", 
                            "Exp.[const.Time]",
                            "Axis[vert.]:Exp.[const.Time]",
                            "v22:Axis[vert.]","v33:Axis[vert.]","vacc:Axis[vert.]","vdec:Axis[vert.]",
                            'Constant'))
###############################################

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
# form <- aSPv ~ 1 + v0*accel*exp + (1 + v0 + accel + exp|sub)
# model <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
#                                                        args=list(control=lmerControl(optimizer='bobyqa'))), data=dataAll)
# formula(model)
aSPv_lmm <- lme(aSPv ~ 1 + v0*accel,
                random = list(sub = ~ 1 + v0 + accel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))


aSPv_lmm_v0 <- lme(aSPv ~ 1 + v0,
                random = list(sub = ~ 1 + v0 ),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_v0)
aSPv_lmm_accel <- lme(aSPv ~ 1 + accel,
                random = list(sub = ~ 1 + accel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=dataAll)
summary(aSPv_lmm_accel)


bicV0   <- BIC(aSPv_lmm_v0)
bicAc   <- BIC(aSPv_lmm_accel)
bicFull <- BIC(aSPv_lmm)

bic_to_bf(c(bicV0, bicAc, bicFull), denominator = bicV0)
bic_to_bf(c(bicV0, bicAc, bicFull), denominator = bicAc)




randomeffects <- data.frame(
             ranef(aSPv_lmm)
  )

colnames(randomeffects) <- c('Intercept', 'v0', 'accel', 'experiment')
v1 <- unique(dataAll$sub)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPv")
fixedeffectsAnti <- data.frame(
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns


write.csv(randomeffects, 'LMM/exp2_cond100Pred_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp2_cond100Pred_lmm_fixedeffectsAnti.csv')



rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "v0", "accel")

starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp2_cond100Pred_lmmResults_antiParams.html', 
                          title='Exp 2A-B: Anticipatory Parameters – Parametric Variables',
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


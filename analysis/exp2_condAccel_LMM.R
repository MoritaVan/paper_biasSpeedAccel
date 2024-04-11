library(nlme)
library(lme4)
library(MASS)
library(stargazer)
library(broom)
library(plyr)
library(buildmer)
library(emmeans)
library(BayesFactor)

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
xVar <- c('sub', 'condition', 'trial', 'exp', 'trial_velocity','sub_txt', 'aSPv_x', 'aSPon_x', 'SPlat_x', 'SPacc_x')
yVar <- c('sub', 'condition', 'trial', 'exp', 'trial_velocity','sub_txt', 'aSPv_y', 'aSPon_y', 'SPlat_y', 'SPacc_y')

setwd("~/Experiments/data/outputs/exp2ctrl/")

dataCtrl <- read.csv('exp2ctrl_params_condAccel.csv', sep=',')
dataCtrl <- dataCtrl[dataCtrl$trial>10,]

dataCtrl$exp <- replicate(nrow(dataCtrl), 'constantTime')


setwd("~/Experiments/data/outputs/exp2/")

data <- read.csv('exp2_params_condAccel.csv', sep=',')
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

new_colnames <- c('sub','condition','trial','exp','trial_velocity','sub_txt','aSPv','aSPon','SPlat','SPacc','axis')
colnames(xAxis) <- new_colnames
colnames(yAxis) <- new_colnames
colnames(xAxisCtrl) <- new_colnames
colnames(yAxisCtrl) <- new_colnames

data <- rbind(xAxis,yAxis,xAxisCtrl,yAxisCtrl)

prob <- data$condition
prob[prob=='Va-100_V0-0'] <- 0.0
prob[prob=='Va-75_Vd-25'] <- 0.30
prob[prob=='Vd-75_Va-25'] <- 0.70
prob[prob=='Vd-100_V0-0'] <- 1.00
prob <- as.numeric(prob)
data$prob <- prob

df <- na.omit(data)
df$trial_velocity <- relevel(as.factor(df$trial_velocity), 'Vd')

aSPv_lmm <- lme(aSPv ~ 1 + prob*exp*axis,
                random = list(sub = ~ 1 + prob + exp + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))

aSPon_lmm <- lme(aSPon ~ 1 + prob*exp*axis,
                 random = list(sub = ~ 1 + prob + exp + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=df)
summary(aSPon_lmm)
qqnorm(aSPon_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPon_lmm))


SPlat_lmm <- lme(SPlat ~ 1 + trial_velocity*prob*exp + trial_velocity*prob*axis,
                 random = list(sub = ~ 1 + prob + exp + axis + trial_velocity),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=df)
summary(SPlat_lmm)
qqnorm(SPlat_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPlat_lmm))


SPacc_lmm <- lme(SPacc ~ 1 + trial_velocity*prob*exp + trial_velocity*prob*axis,
                 random = list(sub = ~ 1 + prob + exp + axis + trial_velocity),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=df)
summary(SPacc_lmm)
qqnorm(SPacc_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPacc_lmm))



randomeffects <- data.frame(
  rbind.fill(ranef(aSPon_lmm),
             ranef(aSPv_lmm),
             ranef(SPlat_lmm),
             ranef(SPacc_lmm)
  ))

colnames(randomeffects) <- c('Intercept', 'prob', 'experiment', 'axis','trial_velocity')
v1 <- unique(df$sub)
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

write.csv(randomeffects, 'exp2_condAccel_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'exp2_condAccel_lmm_fixedeffectsAnti.csv')
write.csv(fixedeffectsVGP, 'exp2_condAccel_lmm_fixedeffectsVGP.csv')




rSA <- ranef(aSPon_lmm)
colnames(rSA) <- c("Constant", "P(V3)", "Exp.[const.time]", 'Axis[vert.]')
rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(V3)", "Exp.[const.time]", 'Axis[vert.]')
rLA <- ranef(SPlat_lmm)
colnames(rLA) <- c("Constant", "P(V3)", "Exp.[const.time]", 'Axis[vert.]', "Trial vel[Va]")
rPA <- ranef(SPacc_lmm)
colnames(rPA) <- c("Constant", "P(V3)", "Exp.[const.time]", 'Axis[vert.]', "Trial vel[Va]")


starAnti.out <- stargazer(aSPon_lmm,aSPv_lmm,
                          out='exp2_condAccel_lmmResults_antiParams.html', 
                          title='Anticipatory Parameters',
                          single.row=FALSE,
                          report = "vc*stp",
                          star.cutoffs = c(.001, .0001, .00001),
                          ci=TRUE, ci.level=0.95, digits=3,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rSA,rAV),
                          dep.var.labels = c("aSPon", "aSPv"),
                          covariate.labels = c(
                            "P(Vd)",
                            "Exp.[const.Time]",
                            "Axis[vert.]",
                            "P(Vd):Exp.[const.Time]",
                            "P(Vd):Axis[vert.]",
                            "Exp.[const.Time]:Axis[vert.]",
                            "P(V3):Exp.[const.Time]:Axis[vert.]",
                            'Constant'))


starVGP.out <- stargazer(SPlat_lmm,SPacc_lmm,
                         out='exp2_condAccel_lmmResults_VGPparams.html', 
                         title='Visually Guided Parameters',
                         single.row=FALSE,
                         report = "vc*stp",
                         star.cutoffs = c(.001, .0001, .00001),
                         ci=TRUE, ci.level=0.95, digits=3,
                         model.numbers = FALSE,
                         omit.stat=c("LL","ser","f", 'aic', 'bic'),
                         keep.stat = c("rsq","f"),
                         add.lines = formatRanef(rLA,rPA),
                         dep.var.labels = c("SPlat", "SPacc"),
                         covariate.labels = c(
                           "Trial vel[Va]",
                           "P(Vd)", 
                           "Exp.[const.Time]",
                           "Axis[vert.]",
                           "Trial vel[Va]:P(Vd)",
                           "Trial vel[Va]:Exp.[const.Time]",
                           
                           "P(Vd):Exp.[const.Time]",
                           "Trial vel[Va]:Axis[vert.]",
                           "P(Vd):Axis[vert.]",
                           "Trial vel[Va]:P(Vd):Exp.[const.Time]",
                           "Trial vel[Va]:P(Vd):Axis[vert.]",
                           'Constant'))

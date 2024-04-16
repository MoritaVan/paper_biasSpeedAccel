library(nlme)
library(lme4)
library(stargazer)
library(broom)
library(tidyverse)
library(MASS)
library(plyr)
library(ggplot2)
library(buildmer)
library(tidyr)

# setwd("../exp2/LMM")
setwd("~/Experiments/data/outputs/exp2")


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


data <- read.csv('exp2_params_condConst.csv', sep=',')
data <- data[data$trial>10,]

prob <- data$condition
prob[prob=='V1-100_V0-0'] <- 0.0
prob[prob=='V1-75_V3-25'] <- 0.30
prob[prob=='V3-75_V1-25'] <- 0.70
prob[prob=='V3-100_V0-0'] <- 1.00
prob <- as.numeric(prob)
data$prob <- prob

n1_tgvel<-data$trial_velocity
n1_tgvel[]<-NaN

for (c in unique(data$condition)) {
  n1_tgvel[data$condition==c][2:nrow(data[data$condition==c,])] <- data[data$condition==c,'trial_velocity'][1:(nrow(data[data$condition==c,])-1)]
}
data$n1_vel <- n1_tgvel

n1_tgvel[n1_tgvel=='V3'] <- 33/sqrt(2)
n1_tgvel[n1_tgvel=='V1'] <- 11/sqrt(2)
n1_tgvel <- as.numeric(n1_tgvel)
data$n1_vel_num <-n1_tgvel

xVar <- c('sub', 'condition', 'trial', 'trial_velocity', 'prob', 'n1_vel', 'n1_vel_num', 'sub_txt', 'aSPv_x', 'aSPon_x', 'SPlat_x', 'SPacc_x')
yVar <- c('sub', 'condition', 'trial', 'trial_velocity', 'prob', 'n1_vel', 'n1_vel_num', 'sub_txt', 'aSPv_y', 'aSPon_y', 'SPlat_y', 'SPacc_y')

xAxis <- data[xVar]
yAxis <- data[yVar]

xAxis$axis <- replicate(nrow(xAxis), 'horiz.')
yAxis$axis <- replicate(nrow(xAxis), 'vert.')

new_colnames <- c('sub','condition','trial','trial_velocity','prob','n1_vel','n1_vel_num','sub_txt','aSPv','aSPon','SPlat','SPacc','axis')
colnames(xAxis) <- new_colnames
colnames(yAxis) <- new_colnames

data <- rbind(xAxis,yAxis)

df <- na.omit(data)
df$n1_vel <- relevel(as.factor(df$n1_vel), 'V3')
df$trial_velocity <- relevel(as.factor(df$trial_velocity), 'V3')


#######################################


aSPv_lmm <- lme(aSPv ~ 1 + prob*axis,
                random = list(sub = ~ 1 + prob  + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))



randomeffects <- data.frame(
             ranef(aSPv_lmm)
  )

colnames(randomeffects) <- c('Intercept', 'prob', 'axis','trial_velocity')
v1 <- unique(df$sub)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPv")
fixedeffectsAnti <- data.frame(
  c1 <- fixef(aSPon_lmm)
)
colnames(fixedeffectsAnti) <- columns


write.csv(randomeffects, 'LMM/exp2_condConst_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp2_condConst_lmm_fixedeffectsAnti.csv')


rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(v33)", 'Axis[vert.]')


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp2_condConst_lmmResults_antiParams.html', 
                          title='Exp 2A: Anticipatory Parameters – Constant Speed Mixture',
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
                            "P(v33)", 
                            "Axis[vert.]",
                            "P(v33):Axis[vert.]",
                            'Constant'))

#########################################

df$prob <- df$prob-0.5
aSPv_lmm <- lme(aSPv ~ 1 + n1_vel*prob,
                random = list(sub = ~ 1 + prob + n1_vel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))


randomeffects <- data.frame(
  ranef(aSPv_lmm)
)

colnames(randomeffects) <- c('Intercept', 'prob', 'n1_tgVel')
v1 <- unique(df$sub)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPon","aSPv")
fixedeffectsAnti <- data.frame(
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns

write.csv(randomeffects, 'LMM/exp2_condConst_lmm_n1Eff_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp2_condConst_lmm_n1Eff_fixedeffectsAnti.csv')

rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(v33)", "N-1 vel[v11]")


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp2_condConst_lmmResults_n1Eff_antiParams.html', 
                          title='Exp 2A: Anticipatory Parameters - Sequential Effects',
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
                            "N-1 vel[v11]",
                            "P(v33)", 
                            "N-1 vel[v11]:P(v33)",
                            'Constant'))

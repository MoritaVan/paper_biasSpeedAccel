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

# setwd("../exp1/LMM")
setwd("~/Experiments/data/outputs/exp1")

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


data <- read.csv('exp1_params.csv', sep=',')
data <- data[data$trial>10,]
data$prob <- data$cond_num

n1_tgvel<-data$trial_vel
n1_tgvel[]<-NaN

for (c in unique(data$cond_num)) {
  n1_tgvel[data$cond_num==c][2:nrow(data[data$cond_num==c,])] <- data[data$cond_num==c,'trial_velocity'][1:(nrow(data[data$cond_num==c,])-1)]
}
data$n1_vel <- n1_tgvel

n1_tgvel[n1_tgvel=='LS'] <- 5.5
n1_tgvel[n1_tgvel=='HS'] <- 16.5
n1_tgvel <- as.numeric(n1_tgvel)
data$n1_vel_num <-n1_tgvel

df <- na.omit(data)



#################


aSPv_lmm <- lme(aSPv ~ 1 + prob,
                random = list(sub = ~ 1 + prob),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))


randomeffects <- data.frame(
             ranef(aSPv_lmm)
  )

colnames(randomeffects) <- c('Intercept', 'prob', 'trial_velocity')
v1 <- unique(df$sub_txt)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPon","aSPv")
fixedeffectsAnti <- data.frame(
  c1 <- fixef(aSPon_lmm),
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns


write.csv(randomeffects, 'LMM/exp1_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp1_lmm_fixedeffectsAnti.csv')

colnames(rAV) <- c("Constant", "P(HS)")


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp1_lmmResults_antiParams.html', 
                          title='Exp1: Anticipatory Parameters',
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
                            "P(HS)", 
                            'Constant'))


#############

aSPv_lmm <- lme(aSPv ~ 1 + n1_vel*prob,
                random = list(sub = ~ 1 + prob + n1_vel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))

df$prob <- df$prob-0.5
aSPv_lmm <- lme(aSPv ~ 1 + n1_vel*prob,
                random = list(sub = ~ 1 + n1_vel + prob),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)


randomeffects <- data.frame(
  rbind.fill(ranef(aSPon_lmm),
             ranef(aSPv_lmm)
  ))

colnames(randomeffects) <- c('Intercept', 'prob', 'n1_tgVel')
v1 <- unique(df$sub_txt)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPv")
fixedeffectsAnti <- data.frame(
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns


write.csv(randomeffects, 'LMM/exp1_lmm_n1Eff_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp1_lmm_n1Eff_fixedeffectsAnti.csv')

rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(HS)", "N-1 vel[LS]")


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp1_lmmResults_n1Eff_antiParams.html', 
                          title='Exp 1: Anticipatory Parameters – Sequential Effects',
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
                            "N-1 vel[LS]",
                            "P(HS)", 
                            "P(HS):N-1 vel[LS]",
                            'Constant'))


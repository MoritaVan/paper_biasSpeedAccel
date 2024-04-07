library(nlme)
library(lme4)
library(stargazer)
library(broom)
library(tidyverse)
library(MASS)
library(plyr)
library(ggplot2)
library(buildmer)


setwd("../exp1/LMM")


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


formatRanef3 <- function(r1, r2, r3) { # receives the ranef structure
  
  sd_r1 <- list()
  sd_r2 <- list()
  sd_r3 <- list()
  coln <- append(colnames(r1),colnames(r2))
  coln <- append(coln,colnames(r3))
  coln <- unique(coln)
  
  listout <- list(
    c('Random Effects', '', '', ''),
    c("Groups", NROW(r1), NROW(r2), NROW(r3))
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
    if (c %in% colnames(r3)){
      sd_r3[c] <- sd(unlist(r3[c]))  
    } else {
      sd_r3[c] <- NaN
    }
    
    listout <- append(listout, list(c(sprintf("sd(%s)",c), sprintf("%.2f",sd_r1[c]), sprintf("%.2f",sd_r2[c]), sprintf("%.2f",sd_r3[c]))))
  }
  listout
}


data <- read.csv('exp1_params.csv', sep=',')

# data$start_anti_x     <- as.numeric(gsub(',', '.', data$start_anti_x))
# data$velocity_model_x <- as.numeric(gsub(',', '.', data$velocity_model_x))
# data$latency_x        <- as.numeric(gsub(',', '.', data$latency_x))
# data$ramp_pursuit_x   <- as.numeric(gsub(',', '.', data$ramp_pursuit_x))
# data$steady_state_x   <- as.numeric(gsub(',', '.', data$steady_state_x))
#data <- na.omit(data)

data <- data[data$trial>50,]

subs  <- unlist(data$sub)
prob  <- unlist(data$cond_num)
tgVel <- unlist(data$trial_vel)
startAnti <- data$start_anti_x
antiVel   <- data$velocity_model_x
latency   <- data$latency_x
purAcc    <- data$ramp_pursuit_x
steadySt  <- data$steady_state_x


lmm_antiOnset <-  lme(startAnti ~ 1 + prob, random = list(subs = ~ 1 + prob), method = 'ML', na.action = na.omit)
summary(lmm_antiOnset)
qqnorm(lmm_antiOnset, ~ resid(., type = "p") | subs, abline = c(0, 1))
hist(resid(lmm_antiOnset))

plot(lmm_antiOnset)


lmm_antiVel <-  lme(antiVel ~ 1 + prob, random = list(subs = ~ 1 + prob), method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(lmm_antiVel)
qqnorm(lmm_antiVel, ~ resid(., type = "p") | subs, abline = c(0, 1))

hist(resid(lmm_antiVel))
plot(lmm_antiVel, col=subs)

plot(latency, col=subs)
plot(purAcc, col=subs)
purAcc[abs(purAcc)>200] <- NaN


# prob and tgvel as main factors

lmm_latency <-  lme(latency ~ 1 + prob*tgVel, random = list(subs = ~ 1 + prob + tgVel), method = 'ML', na.action = na.omit)
summary(lmm_latency)
qqnorm(lmm_latency, ~ resid(., type = "p") | subs, abline = c(0, 1))
hist(resid(lmm_latency))
plot(lmm_latency, col=subs)

lmm_purAcc <-  lme(purAcc ~ 1 + prob*tgVel, random = list(subs = ~ 1 + prob + tgVel), method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(lmm_purAcc)
qqnorm(lmm_purAcc, ~ resid(., type = "p") | subs, abline = c(0, 1))
hist(resid(lmm_purAcc))
plot(lmm_purAcc, col=subs)

lmm_steadySt <-  lme(steadySt ~ 1 + prob*tgVel, random = list(subs = ~ 1 + prob + tgVel), method = 'ML', na.action = na.omit)
summary(lmm_steadySt)
qqnorm(lmm_steadySt, ~ resid(., type = "p") | subs, abline = c(0, 1))
hist(resid(lmm_steadySt))


randomeffects <- data.frame(
  rbind.fill(ranef(lmm_antiOnset),
        ranef(lmm_antiVel),
        ranef(lmm_latency),
        ranef(lmm_purAcc),
        ranef(lmm_steadySt)
))

colnames(randomeffects) <- c('Intercept', 'prob', 'tgVel')
v1 <- unique(subs)
randomeffects$sub <- c(v1,v1,v1,v1,v1)
randomeffects$var <- c(rep("start_anti_x", length(v1)), rep("velocity_model_x", length(v1)), rep("latency_x", length(v1)), rep("ramp_pursuit_x", length(v1)), rep("steady_state_x", length(v1)))

columns = c("start_anti_x","velocity_model_x")
fixedeffectsAnti <- data.frame(
  startAnti <- fixef(lmm_antiOnset),
  antiVel <- fixef(lmm_antiVel)
)
colnames(fixedeffectsAnti) <- columns

columns = c("latency_x","ramp_pursuit_x","steady_state_x")
fixedeffectsVGP <- data.frame(
  latency <- fixef(lmm_latency),
  purAcc <- fixef(lmm_purAcc),
  stState <- fixef(lmm_steadySt)
)
colnames(fixedeffectsVGP) <- columns

write.csv(randomeffects, 'exp1_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'exp1_lmm_fixedeffectsAnti.csv')
write.csv(fixedeffectsVGP, 'exp1_lmm_fixedeffectsVGP.csv')


rSA <- ranef(lmm_antiOnset)
colnames(rSA) <- c("Constant", "P(HS)")
rAV <- ranef(lmm_antiVel)
colnames(rAV) <- c("Constant", "P(HS)")
rLA <- ranef(lmm_latency)
colnames(rLA) <- c("Constant", "P(HS)", "Target Velocity")
rPA <- ranef(lmm_purAcc)
colnames(rPA) <- c("Constant", "P(HS)", "Target Velocity")
rSS <- ranef(lmm_steadySt)
colnames(rSS) <- c("Constant", "P(HS)", "Target Velocity")


starAnti.out <- stargazer(lmm_antiOnset,lmm_antiVel,
                          out='exp1_LMM_antiParams_drop50.html', title='Best Linear Mixed Effects Model - Anticipatory Parameters',
                          single.row=TRUE,
                          report = "vc*st",
                          star.cutoffs = c(.05, .01, .001),
                          ci=TRUE, ci.level=0.95, digits=2,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rSA,rAV),
                          dep.var.labels = c("aSPon", "aSPv"),
                          covariate.labels = c("P(HS)", 'Constant'))


starVGP.out <- stargazer(lmm_latency,lmm_purAcc,lmm_steadySt,
                         out='exp1_LMM_VGPparams_drop50.html', title='Best Linear Mixed Effects Model - Visually Guided Parameters',
                         single.row=TRUE,
                         report = "vc*st",
                         star.cutoffs = c(.05, .01, .001),
                         ci=TRUE, ci.level=0.95, digits=2,
                         model.numbers = FALSE,
                         omit.stat=c("LL","ser","f", 'aic', 'bic'),
                         keep.stat = c("rsq","f"),
                         add.lines = formatRanef3(rLA,rPA,rSS),
                         dep.var.labels = c("SPlat", "SPacc", "SPss"),
                         covariate.labels = c("P(HS)","TargetVelocity [LS]",'P(HS):TargetVelocity','Constant'))


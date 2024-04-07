library(nlme)
library(lme4)
library(MASS)
library(stargazer)
library(broom)
library(plyr)
library(buildmer)

######################################
### Cond Acc, all data together

var2keep <- c('cond', 'trial', 'trial_vel', 'start_anti_x', 'start_anti_y', 
              'ramp_pursuit_x', 'ramp_pursuit_y', 'latency_x', 'latency_y',
              'sub', 'vel_x', 'vel_y')


setwd("~/Experiments/biasAcceleration/stats/LMM_biasAcceleration/")

tmp_dist <- read.csv('dadosANEMO_allSubs_condAcc.csv', sep=',')
tmp_dist <- tmp_dist[var2keep]
colnames(tmp_dist)[which(names(tmp_dist) == 'ramp_pursuit_x')] <- 'accel_pursuit_x'
colnames(tmp_dist)[which(names(tmp_dist) == 'ramp_pursuit_y')] <- 'accel_pursuit_y'
#tmp_dist <- na.omit(tmp_dist)


# tmp_dist$dist <- replicate(nrow(tmp_dist),0) # 0 is the reference level for the lme
# tmp_dist$time <- replicate(nrow(tmp_dist),1)
tmp_dist$exp <- replicate(nrow(tmp_dist), 'constantDistance')

maxSub <- max(tmp_dist$sub)

var2keep <- c('cond', 'trial', 'trial_vel', 'start_anti_x', 'start_anti_y', 
              'accel_pursuit_x', 'accel_pursuit_y', 'latency_x', 'latency_y',
              'sub', 'vel_x', 'vel_y')

setwd("~/Experiments/biasAccelerationControl/stats/LMM_biasAccelerationControl/")


tmp_time <- read.csv('dadosANEMO_allSubs_condAcc.csv', sep=',')
tmp_time <- tmp_time[var2keep]
#tmp_acc <- na.omit(tmp_acc)
# tmp_time$dist <- replicate(nrow(tmp_time),1)
# tmp_time$time <- replicate(nrow(tmp_time),0)
tmp_time$exp <- replicate(nrow(tmp_time), 'constantDistance')
tmp_time$sub <- tmp_time$sub + maxSub
unique(tmp_time$sub)
tmp_time[tmp_time$sub==maxSub+1,'sub'] <- 1
tmp_time[tmp_time$sub==maxSub+2,'sub'] <- 13
unique(tmp_time$sub)

data <- rbind(tmp_dist,tmp_time)
data <- data[data$trial>50,]

dataAcc <- rbind(tmp_dist,tmp_time)

subs  <- c(data$sub,data$sub)
tgvel <- c(data$trial_vel,data$trial_vel)
prob  <- c(data$cond,data$cond)
startAnti <- c(data$start_anti_x,data$start_anti_y)
antiVel   <- c(data$vel_x,data$vel_y)
latency   <- c(data$latency_x,data$latency_y)
purAcc    <- c(data$accel_pursuit_x,data$accel_pursuit_y)
exp <- c(data$exp,data$exp)
x<-data$sub
x[]<- 'x'
y<-data$sub
y[]<- 'y'
axis <- c(x,y)

prob[prob=='Va-100_V0-0'] <- 0.0
prob[prob=='Va-75_Vd-25'] <- 0.3
prob[prob=='Vd-75_Va-25'] <- 0.7
prob[prob=='Vd-100_V0-0'] <- 1.0
prob <- as.numeric(prob)

tgvel <- relevel(as.factor(tgvel), 'Vd')

######## Anticip vel: not significant

form <- antiVel ~ 1 + prob*exp*axis + (1 + prob + axis + exp|subs)
mantiVel <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                          args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mantiVel)

antiVel_x_full <- lme(antiVel ~ 1 + prob + axis,
                      random = list(subs = ~ 1 + prob + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)


######## Start anticipation: not significant

form <- startAnti ~ 1 + prob*exp*axis + (1 + prob + exp + axis|subs)
mstartAnti <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                            args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mstartAnti)
startAnti_x_full <- lme(startAnti ~  1 + axis + prob, 
                        random = list(subs = ~ 1 + axis + prob),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)

####### Latency

form <- latency ~ 1 + prob*exp*axis + prob*tgvel*exp + (1 + prob + exp + axis + tgvel|subs)
mlatency <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                          args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mlatency)
latency_x_full <- lme(latency ~ 1 + axis + prob + axis:prob + tgvel + prob:tgvel, 
                      random = list(subs = ~ 1 + tgvel + prob + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)

latency_x_full <- lme(latency ~ 1 + axis + tgvel + prob + tgvel:prob + axis:prob, 
                      random = list(subs = ~ 1 + tgvel + prob + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)

####### Pursuit acceleration

form <- purAcc ~ 1 + prob*exp*axis + prob*tgvel*exp + (1 + prob + exp + axis + tgvel|subs)
mpurAcc <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                         args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mpurAcc)

# dropping first 50 trials
purAcc_x_full <- lme(purAcc ~ 1 + tgvel + axis + prob + axis:prob, 
                     random = list(subs = ~ 1 + tgvel + prob + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)


rSA <- ranef(startAnti_x_full)
colnames(rSA) <- c("Constant", "Axis", "P(Vd)")
rAV <- ranef(antiVel_x_full)
colnames(rAV) <- c("Constant", "P(Vd)", "Axis")
rLA <- ranef(latency_x_full)
colnames(rLA) <- c("Constant", "Target Velocity" ,"P(Vd)", "Axis")
rPA <- ranef(purAcc_x_full)
colnames(rPA) <- c("Constant", "Target Velocity" ,"P(Vd)", "Axis")


## dropping 50 trials

starAnti.out <- stargazer(startAnti_x_full,antiVel_x_full,
                          out='maxLMM_biasAccel_condAcc_DistVsTime_antiParams_drop50.html', title='Best Linear Mixed Effects Model - Anticipatory Parameters',
                          single.row=TRUE,
                          report = "vc*st",
                          star.cutoffs = c(.05, .01, .001),
                          ci=TRUE, ci.level=0.95, digits=2,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rSA,rAV),
                          dep.var.labels = c("aSPon", "aSPv"))#,
covariate.labels =  c(
  "Axis[y]",
  "P(Vd)",
  'Constant'))

starVGP.out <- stargazer(latency_x_full,purAcc_x_full,
                         out='maxLMM_biasAccel_condAcc_DistVsTime_VGPparams_drop50.html', title='Best Linear Mixed Effects Model - Visually Guided Parameters',
                         single.row=TRUE,
                         report = "vc*st",
                         star.cutoffs = c(.05, .01, .001),
                         ci=TRUE, ci.level=0.95, digits=2,
                         model.numbers = FALSE,
                         omit.stat=c("LL","ser","f", 'aic', 'bic'),
                         keep.stat = c("rsq","f"),
                         add.lines = formatRanef(rLA,rPA),
                         dep.var.labels = c("SPlat", "SPacc"))#,
covariate.labels =  c(
  "Axis[y]",
  "TargetVelocity [Va]",
  "P(Vd)",
  "P(Vd):TargetVelocity",
  "P(Vd):Axis",
  'Constant'))






columns = c("start_anti","vel","latency","ramp_pursuit")
randomeffects <- data.frame(
  rbind.fill(ranef(startAnti_x_full),
             ranef(antiVel_x_full),
             ranef(latency_x_full),
             ranef(purAcc_x_full)
  ))
colnames(randomeffects) <- c('Intercept', 'axisy', 'prob', 'tgvel')
v1 <- unique(subs)
randomeffects$sub <- c(v1,v1,v1,v1)
randomeffects$var <- c(rep("start_anti", length(v1)), rep("vel", length(v1)), rep("latency", length(v1)), rep("ramp_pursuit", length(v1)))


fixedeffectsAntiVel <- data.frame(
  antiVel <- fixef(antiVel_x_full)
)
colnames(fixedeffectsAntiVel) <- 'vel'

fixedeffectsStartAnti <- data.frame(
  startAnti <- fixef(startAnti_x_full)
)
colnames(fixedeffectsStartAnti) <- 'start_anti'

fixedeffectsLat <- data.frame(
  latency <- fixef(latency_x_full)
)
colnames(fixedeffectsLat) <- 'latency'

fixedeffectsPurAcc <- data.frame(
  purAcc <- fixef(purAcc_x_full)
)
colnames(fixedeffectsPurAcc) <- "ramp_pursuit"

write.csv(randomeffects, 'lme_biasAccel_condAcc_randomEffects.csv')
write.csv(fixedeffectsAntiVel, 'lme_biasAccel_condAcc_fixedEffectsAntiVel.csv')
write.csv(fixedeffectsStartAnti, 'lme_biasAccel_condAcc_fixedEffectsStartAnti.csv')
write.csv(fixedeffectsLat, 'lme_biasAccel_condAcc_fixedEffectsLatency.csv')
write.csv(fixedeffectsPurAcc, 'lme_biasAccel_condAcc_fixedEffectsPurAcc.csv')


library(nlme)
library(lme4)
library(MASS)
library(stargazer)
library(broom)
library(plyr)
library(buildmer)

var2keep <- c('cond', 'trial', 'trial_vel', 'start_anti_x', 'start_anti_y', 
              'ramp_pursuit_x', 'ramp_pursuit_y', 'latency_x', 'latency_y',
              'sub', 'vel_x', 'vel_y')

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
#######################################################
#####################################################################################
### Cond 100, all data together

var2keep <- c('cond', 'trial', 'trial_vel', 'start_anti_x', 'start_anti_y', 
              'ramp_pursuit_x', 'ramp_pursuit_y', 'latency_x', 'latency_y',
              'sub', 'vel_x', 'vel_y')

setwd("~/Experiments/biasAcceleration/stats/LMM_biasAcceleration/")

tmp_dist <- read.csv('dadosANEMO_allSubs_cond100.csv', sep=',')
tmp_dist <- tmp_dist[var2keep]
colnames(tmp_dist)[which(names(tmp_dist) == 'ramp_pursuit_x')] <- 'accel_pursuit_x'
colnames(tmp_dist)[which(names(tmp_dist) == 'ramp_pursuit_y')] <- 'accel_pursuit_y'

tmp_dist$exp <- replicate(nrow(tmp_dist), 'constantDistance')
maxSub <- max(tmp_dist$sub)

var2keep <- c('cond', 'trial', 'trial_vel', 'start_anti_x', 'start_anti_y', 
              'accel_pursuit_x', 'accel_pursuit_y', 'latency_x', 'latency_y',
              'sub', 'vel_x', 'vel_y')

setwd("~/Experiments/biasAccelerationControl/stats/LMM_biasAccelerationControl/")

tmp_time <- read.csv('dadosANEMO_allSubs_cond100.csv', sep=',')
tmp_time <- tmp_time[var2keep]
tmp_time$exp <- replicate(nrow(tmp_time), 'constantTime')
tmp_time$sub <- tmp_time$sub + maxSub
unique(tmp_time$sub)
tmp_time[tmp_time$sub==maxSub+1,'sub'] <- 1
tmp_time[tmp_time$sub==maxSub+2,'sub'] <- 13
unique(tmp_time$sub)

data <- rbind(tmp_dist,tmp_time)
data100 <- rbind(tmp_dist,tmp_time)

subs  <- c(data$sub,data$sub)
tgvel <- c(data$trial_vel,data$trial_vel)
cond  <- c(data$cond,data$cond)
startAnti <- c(data$start_anti_x,data$start_anti_y)
antiVel   <- c(data$vel_x,data$vel_y)
latency   <- c(data$latency_x,data$latency_y)
purAcc    <- c(data$accel_pursuit_x,data$accel_pursuit_y)

exp <- c(data$exp,data$exp)

# create dummy variable axis
x<-data$sub
x[]<- 'x'
y<-data$sub
y[]<- 'y'
axis <- c(x,y)

cond[cond=='V1-100_V0-0'] <- 'V1'
cond[cond=='V2-100_V0-0'] <- 'V2'
cond[cond=='V3-100_V0-0'] <- 'V3'
cond[cond=='Va-100_V0-0'] <- 'Va'
cond[cond=='Vd-100_V0-0'] <- 'Vd'

cVa <- relevel(as.factor(cond), 'Va')
cVd <- relevel(as.factor(cond), 'Vd')
cV2 <- relevel(as.factor(cond), 'V2')
cV3 <- relevel(as.factor(cond), 'V3')
######## Anticip vel: 

#### exp
form <- antiVel ~ 1 + cond*exp + cond*axis + exp*axis + (1 + cond + axis + exp|subs)
form <- antiVel ~ 1 + cond*exp*axis + (1 + cond + axis + exp|subs)
mantiVel <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                          args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mantiVel)
antiVel_x_full <- lme(antiVel ~ 1 + cond + axis + exp + axis:exp + cond:axis + cond:exp, 
                      random = list(subs = ~ 1 + cond+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)

antiVel_x_full <- lme(antiVel ~ 1 + cVa + axis + exp + axis:exp + cVa:axis + cVa:exp, 
                      random = list(subs = ~ 1 + cVa+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)
antiVel_x_full <- lme(antiVel ~ 1 + cVd + axis + exp + axis:exp + cVd:axis + cVd:exp, 
                      random = list(subs = ~ 1 + cVd+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)
antiVel_x_full <- lme(antiVel ~ 1 + cV2 + axis + exp + axis:exp + cV2:axis + cV2:exp, 
                      random = list(subs = ~ 1 + cV2+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)
antiVel_x_full <- lme(antiVel ~ 1 + cV3 + axis + exp + axis:exp + cV3:axis + cV3:exp, 
                      random = list(subs = ~ 1 + cV3+axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(antiVel_x_full)


######## Start anticipation: 

#### exp
form <- startAnti ~ 1 + cond*exp + cond*axis + exp*axis + (1 + cond + exp + axis|subs)
form <- startAnti ~ 1 + cond*exp*axis + (1 + cond + exp + axis|subs)
mstartAnti <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                            args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mstartAnti)
startAnti_x_full <- lme(startAnti ~  1 + cond + axis, random = list(subs = ~ 1 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)

startAnti_x_full <- lme(startAnti ~  1 + cVa + axis, random = list(subs = ~ 1 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)
startAnti_x_full <- lme(startAnti ~  1 + cVd + axis, random = list(subs = ~ 1 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)
startAnti_x_full <- lme(startAnti ~  1 + cV2 + axis, random = list(subs = ~ 1 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)
startAnti_x_full <- lme(startAnti ~  1 + cV3 + axis, random = list(subs = ~ 1 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(startAnti_x_full)

####### Latency: 
##### exp
form <- latency ~ 1 + cond*exp + cond*axis + exp*axis + (1 + cond + exp + axis|subs)
form <- latency ~ 1 + cond*exp*axis + (1 + cond + exp + axis|subs)
mlatency <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                          args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mlatency)
latency_x_full <- lme(latency ~ cond + exp + axis + cond:axis + cond:exp, 
                      random = list(subs = ~ 1 + cond + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)


latency_x_full <- lme(latency ~ cVa + exp + axis + cVa:axis + cVa:exp, 
                      random = list(subs = ~ 1 + cVa + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)
latency_x_full <- lme(latency ~ cVd + exp + axis + cVd:axis + cVd:exp, 
                      random = list(subs = ~ 1 + cVd + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)
latency_x_full <- lme(latency ~ cV2 + exp + axis + cV2:axis + cV2:exp, 
                      random = list(subs = ~ 1 + cV2 + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)
latency_x_full <- lme(latency ~ cV3 + exp + axis + cV3:axis + cV3:exp, 
                      random = list(subs = ~ 1 + cond + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(latency_x_full)

####### Pursuit acceleration: sig effect, cond->dist >> cond->time
##### exp
form <- purAcc ~ 1 + cond*exp + cond*axis + exp*axis + (1 + cond + axis + exp|subs)
form <- purAcc ~ 1 + cond*exp*axis + (1 + cond + axis + exp|subs)
mpurAcc <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
                                                         args=list(control=lmerControl(optimizer='bobyqa'))))
formula(mpurAcc)
purAcc_x_full <- lme(purAcc ~ 1 + cond + exp + cond:exp + axis + cond:axis, 
                     random = list(subs = ~ 1+ axis + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)

# initial cond*exp*axis
purAcc_x_full <- lme(purAcc ~ 1 + cond + exp + cond:exp + axis + cond:axis + exp:axis + 
                       cond:exp:axis, 
                     random = list(subs = ~ 1+ axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)

purAcc_x_full <- lme(purAcc ~ 1 + cVa + exp + cVa:exp + axis + cVa:axis + exp:axis + 
                       cVa:exp:axis, 
                     random = list(subs = ~ 1+ axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)
purAcc_x_full <- lme(purAcc ~ 1 + cVd + exp + cVd:exp + axis + cVd:axis + exp:axis + 
                       cVd:exp:axis, 
                     random = list(subs = ~ 1+ axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)
purAcc_x_full <- lme(purAcc ~ 1 + cV2 + exp + cV2:exp + axis + cV2:axis + exp:axis + 
                       cV2:exp:axis, 
                     random = list(subs = ~ 1+ axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)
purAcc_x_full <- lme(purAcc ~ 1 + cV3 + exp + cV3:exp + axis + cV3:axis + exp:axis + 
                       cV3:exp:axis, 
                     random = list(subs = ~ 1+ axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"))
summary(purAcc_x_full)

rSA <- ranef(startAnti_x_full)
colnames(rSA) <- c("Constant", "Axis")
rAV <- ranef(antiVel_x_full)
colnames(rAV) <- c("Constant", "V2", "V3", "Va", "Vd", "Axis")
rLA <- ranef(latency_x_full)
colnames(rLA) <- c("Constant", "V2", "V3", "Va", "Vd", 'Axis')
rPA <- ranef(purAcc_x_full)
colnames(rPA) <- c("Constant", 'Axis')

starAnti.out <- stargazer(startAnti_x_full,antiVel_x_full,
                          # out='maxLMM_biasAccelCtrl_cond100_DistVsTime_antiParams.html', title='Best Linear Mixed Effects Model - Anticipatory Parameters',
                          out='maxLMM_biasAccelCtrl_cond100_DistVsTime_antiParams_V2-BaseLevel.html', title='Best Linear Mixed Effects Model - Anticipatory Parameters',
                          single.row=TRUE,
                          report = "vc*st",
                          star.cutoffs = c(.05, .01, .001),
                          ci=TRUE, ci.level=0.95, digits=2,
                          model.numbers = FALSE,
                          omit.stat=c("LL","ser","f", 'aic', 'bic'),
                          keep.stat = c("rsq","f"),
                          add.lines = formatRanef(rSA,rAV),
                          dep.var.labels = c("aSPon", "aSPv"))#,
# covariate.labels =  c(
#   "V2","V3","Va","Vd",
#   "Axis[y]",
#   "Experiment [constant time]",
#   "Axis:Experiment",
#   "V2:Axis","V3:Axis","Va:Axis","Vd:Axis",
#   "V2:Experiment","V3:Experiment","Va:Experiment","Vd:Experiment",
#   'Constant'))


starVGP.out <- stargazer(latency_x_full,purAcc_x_full,
                         # out='maxLMM_biasAccelCtrl_cond100_DistVsTime_VGPparams.html', title='Best Linear Mixed Effects Model - Visually Guided Parameters',
                         out='maxLMM_biasAccelCtrl_cond100_DistVsTime_VGPparams_V3-BaseLevel.html', title='Best Linear Mixed Effects Model - Visually Guided Parameters',
                         single.row=TRUE,
                         report = "vc*st",
                         star.cutoffs = c(.05, .01, .001),
                         ci=TRUE, ci.level=0.95, digits=2,
                         model.numbers = FALSE,
                         omit.stat=c("LL","ser","f", 'aic', 'bic'),
                         keep.stat = c("rsq","f"),
                         add.lines = formatRanef(rLA,rPA),
                         dep.var.labels = c("SPlat", "SPacc"))#,
# covariate.labels =  c(
#                       "V2","V3","Va","Vd",
#                       "Experiment [constant time]",
#                       "Axis[y]",
#                       "V2:Axis","V3:Axis","Va:Axis","Vd:Axis",
#                       "Axis:Experiment",
#                       "V2:Experiment:Axis","V3:Experiment:Axis","Va:Experiment:Axis","Vd:Experiment:Axis",
#                       "V2:Experiment","V3:Experiment","Va:Experiment","Vd:Experiment",
#                       'Constant'))


######################

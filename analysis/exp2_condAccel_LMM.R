library(nlme)
library(lme4)
library(MASS)
library(stargazer)
library(broom)
library(plyr)
library(buildmer)
library(emmeans)
library(BayesFactor)


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

# form <- aSPv ~ 1 + prob*exp*axis + (1 + prob + exp + axis|sub)
# model <- buildmer(form,buildmerControl=buildmerControl(direction=c('order','backward'),
#                                                        args=list(control=lmerControl(optimizer='bobyqa'))), data=df)
# formula(model)
aSPv_lmm <- lme(aSPv ~ 1 + prob + axis + exp + axis:exp,
                random = list(sub = ~ 1 + prob + axis),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))

# aSPv_lmm <- lme(aSPv ~ 1 + prob*axis*exp,
#                 random = list(sub = ~ 1 + prob + axis + exp),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
#                 data=df)
# summary(aSPv_lmm)


randomeffects <- data.frame(
             ranef(aSPv_lmm)
  )

colnames(randomeffects) <- c('Intercept', 'prob', 'experiment', 'axis','trial_velocity')
v1 <- unique(df$sub)
randomeffects$sub <- v1
randomeffects$var <- rep("aSPv", length(v1))

columns = c("aSPv")
fixedeffectsAnti <- data.frame(
  c2 <- fixef(aSPv_lmm)
)
colnames(fixedeffectsAnti) <- columns

write.csv(randomeffects, 'LMM/exp2_condAccel_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp2_condAccel_lmm_fixedeffectsAnti.csv')


rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(V3)",'Axis[vert.]')


starAnti.out <- stargazer(aSPv_lmm,
                          out='LMM/exp2_condAccel_lmmResults_antiParams.html', 
                          title='Exp 2A-B: Anticipatory Parameters – Accelerating Target',
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
                            "P(vdec)",
                            "Axis[vert.]",
                            "Exp.[const.Time]",
                            
                            "Exp.[const.Time]:Axis[vert.]",
                            'Constant'))

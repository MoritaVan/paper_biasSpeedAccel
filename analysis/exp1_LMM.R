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

aSPv_lmm <- lme(aSPv ~ 1 + n1_vel*prob,
                random = list(sub = ~ 1 + prob + n1_vel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPv_lmm)
qqnorm(aSPv_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPv_lmm))

aSPon_lmm <- lme(aSPon ~ 1 + n1_vel*prob,
                random = list(sub = ~ 1 + prob + n1_vel),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                data=df)
summary(aSPon_lmm)
qqnorm(aSPon_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(aSPon_lmm))


SPlat_lmm <- lme(SPlat ~ 1 + trial_velocity*prob,
                 random = list(sub = ~ 1 + prob + trial_velocity),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=df)
summary(SPlat_lmm)
qqnorm(SPlat_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPlat_lmm))


SPacc_lmm <- lme(SPacc ~ 1 + trial_velocity*prob,
                 random = list(sub = ~ 1 + prob + trial_velocity),method = 'ML', na.action = na.omit, control = lmeControl(opt = "optim"),
                 data=df)
summary(SPacc_lmm)
qqnorm(SPacc_lmm, ~ resid(., type = "p") | sub, abline = c(0, 1))
hist(resid(SPacc_lmm))


df_plot <- df

df_plot$n1_vel <- as.factor(df_plot$n1_vel)
df_plot$prob <- as.factor(df_plot$prob)

p1 <- ggplot(df_plot, aes(x=prob, y=aSPv, fill=n1_vel)) + 
  geom_boxplot()

randomeffects <- data.frame(
  rbind.fill(ranef(aSPon_lmm),
             ranef(aSPv_lmm),
             ranef(SPlat_lmm),
             ranef(SPacc_lmm)
  ))

colnames(randomeffects) <- c('Intercept', 'prob', 'n1_tgVel', 'trial_velocity')
v1 <- unique(df$sub_txt)
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

write.csv(randomeffects, 'LMM/exp1_lmm_randomEffects.csv')
write.csv(fixedeffectsAnti, 'LMM/exp1_lmm_fixedeffectsAnti.csv')
write.csv(fixedeffectsVGP, 'LMM/exp1_lmm_fixedeffectsVGP.csv')




rSA <- ranef(aSPon_lmm)
colnames(rSA) <- c("Constant", "P(HS)", "N-1 vel[LS]")
rAV <- ranef(aSPv_lmm)
colnames(rAV) <- c("Constant", "P(HS)", "N-1 vel[LS]")
rLA <- ranef(SPlat_lmm)
colnames(rLA) <- c("Constant", "P(HS)", "Target Vel[LS]")
rPA <- ranef(SPacc_lmm)
colnames(rPA) <- c("Constant", "P(HS)", "Target Vel[LS]")


starAnti.out <- stargazer(aSPon_lmm,aSPv_lmm,
                          out='LMM/exp1_lmmResults_antiParams.html', 
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
                                              "N-1 vel[LS]",
                                              "P(HS)", 
                                              "P(HS):N-1 vel[LS]",
                                              'Constant'))


starVGP.out <- stargazer(SPlat_lmm,SPacc_lmm,
                         out='LMM/exp1_lmmResults_VGPparams.html', 
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
                           "Target Vel[LS]",
                           "P(HS)",
                           'P(HS):Target Vel[LS]',
                           'Constant'))

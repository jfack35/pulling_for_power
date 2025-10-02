library(arrow)
library(fdaMixed)
library(tidyverse)

all_swings <- bind_rows(
  read_parquet("left_inside_pull_ik/left_inside_pull_ik_math.parquet")  %>% mutate(handedness = "L", pitch_loc = "inside"),
  read_parquet("right_inside_pull_ik/right_inside_pull_ik_math.parquet") %>% mutate(handedness = "R", pitch_loc = "inside"),
  read_parquet("left_outside_pull_ik/left_outside_pull_ik_math.parquet") %>% mutate(handedness = "L", pitch_loc = "outside"),
  read_parquet("right_outside_pull_ik/right_outside_pull_ik_math.parquet") %>% mutate(handedness = "R", pitch_loc = "outside")
)

all_swings <- all_swings %>%
  mutate(
    pitch_loc = factor(pitch_loc),
    handedness = factor(handedness),
    player = factor(player),
    play_guid = factor(play_guid),
    axis = replace_na(axis, 'k')
  )

### MAKING A LOOP

combos <- all_swings %>% 
  select(spray, joint, feature_type, axis) %>% 
  distinct() %>% 
  mutate(axis = replace_na(axis, 'k')) #k is the angular velocity axis, just so it isn't zero

dir.create("fda_models", showWarnings = FALSE)

fda_significance <- data.frame(
  spray = character(),
  joint = character(),
  feature_type = character(), 
  axis = character(),
  t_stat = numeric(),
  dof = numeric(),
  p_value = numeric(),
  random_effect_var = numeric(),
  residual_var = numeric(),
  stringsAsFactors = FALSE  
)


for (i in 1:nrow(combos)) {
  row <- combos[i, ]
  
  spray1 <- row$spray
  joint1 <- row$joint
  feature_type1 <- row$feature_type
  axis1 <- row$axis

  # spray1 <- "pull"
  # joint1 <- "L_Knee"
  # feature_type1 <- "angular_acceleration"
  # axis1 <- "k"
  
  df <- all_swings %>%
    filter(spray == spray1, joint == joint1, feature_type == feature_type1, axis == axis1) %>% 
    select(play_guid, frame, player, horizontal, bat_side, value)
  
  #getting the number of frames with the greatest sample size
  frame_counts <- table(df$play_guid)       # number of frames per play
  count_freqs <- table(frame_counts)                 # frequency of each frame count
  most_frames <- as.numeric(names(count_freqs)[which.max(count_freqs)])
  
  valid_ids <- names(which(table(df$play_guid) == most_frames))
  df <- df %>% filter(play_guid %in% valid_ids)
  
  #design of the model
  #horizontal is fixed effects, player and handedness are random effects
  design <- df %>% 
    select(horizontal, player, bat_side)
  
  fit <- fdaLm(value | play_guid ~ horizontal | player + bat_side, 
               data = df, 
               design = design,
               nlSearch = FALSE)
  
  #saving the model
  model_name <- paste(spray1, joint1, feature_type1, axis1, sep = "_")
  file_path <- paste0("fda_models/", model_name, "_model.rds")
  
  saveRDS(fit, file = file_path)
  
  print("saved model")
  
  #significance
  t_stat <- fit$betaHat[2] / sqrt(fit$betaVar[2, 2])
  dof <- length(unique(df$play_guid)) - length(fit$betaHat)
  p_value <- 2 * (1 - pt(abs(t_stat), dof))
  
  #variance
  Ghat <- fit$Ghat       # Random effect variance structure
  sigma2hat <- fit$sigma2hat
  total_var <- sum(Ghat) + sigma2hat
  random_effect_var <- sum(Ghat) / total_var
  residual_var <- sigma2hat / total_var
  
  new_row <- data.frame(
    spray = spray1,
    joint = joint1,
    feature_type = feature_type1, 
    axis = axis1,
    t_stat = t_stat, 
    dof = dof,
    p_value = p_value,
    random_effect_var = random_effect_var,
    residual_var = residual_var
  )
  
  fda_significance <- rbind(fda_significance, new_row)
  write.csv(fda_significance, "fda_significance.csv", row.names = FALSE)
  
  print("saved significance csv")
  
  #saving for graphs
  play_ids <- unique(df$play_guid)
  
  file_name <- paste0("fda_models/", spray1, "_", joint1, "_", feature_type1, "_", axis1, "_features")
  
  dir.create(file_name, showWarnings = FALSE)
  
  for (i in seq_along(play_ids)) {
    pid <- play_ids[i]
    df_sub <- df %>% filter(play_guid == pid)
    
    start_idx <- (i - 1) * most_frames + 1
    end_idx <- i * most_frames
    xblup_sub <- fit$xBLUP[start_idx:end_idx]
    
    df_sub$wrist_velocity <- xblup_sub
    
    saveRDS(df_sub, file = paste0(file_name, "/", pid, ".rds"))
  }
  
  print(paste(i, "/", nrow(combos), "are done"))
}  

import os
from PIL import Image, ImageOps

os.makedirs("results/diagrams", exist_ok=True)

def combine_horizontal(img_path1, img_path2, output_path, title1="", title2="", bg_color=(7, 24, 45)):
    im1 = Image.open(img_path1).convert("RGBA")
    im2 = Image.open(img_path2).convert("RGBA")
    
    # Normalize heights
    target_h = max(im1.height, im2.height)
    w1 = int(im1.width * (target_h / im1.height))
    w2 = int(im2.width * (target_h / im2.height))
    im1_r = im1.resize((w1, target_h), Image.Resampling.LANCZOS)
    im2_r = im2.resize((w2, target_h), Image.Resampling.LANCZOS)
    
    gutter = 30
    total_w = w1 + w2 + gutter
    
    new_im = Image.new("RGBA", (total_w, target_h), bg_color)
    new_im.paste(im1_r, (0, 0), im1_r if im1_r.mode == "RGBA" else None)
    new_im.paste(im2_r, (w1 + gutter, 0), im2_r if im2_r.mode == "RGBA" else None)
    
    new_im.save(output_path)
    print(f"Saved horizontal composite: {output_path} ({total_w}x{target_h})")

def combine_vertical(img_path1, img_path2, output_path, bg_color=(7, 24, 45)):
    im1 = Image.open(img_path1).convert("RGBA")
    im2 = Image.open(img_path2).convert("RGBA")
    
    target_w = max(im1.width, im2.width)
    h1 = int(im1.height * (target_w / im1.width))
    h2 = int(im2.height * (target_w / im2.width))
    im1_r = im1.resize((target_w, h1), Image.Resampling.LANCZOS)
    im2_r = im2.resize((target_w, h2), Image.Resampling.LANCZOS)
    
    gutter = 25
    total_h = h1 + h2 + gutter
    
    new_im = Image.new("RGBA", (target_w, total_h), bg_color)
    new_im.paste(im1_r, (0, 0), im1_r if im1_r.mode == "RGBA" else None)
    new_im.paste(im2_r, (0, h1 + gutter), im2_r if im2_r.mode == "RGBA" else None)
    
    new_im.save(output_path)
    print(f"Saved vertical composite: {output_path} ({target_w}x{total_h})")

# 1. PV Curves Irradiance & Temperature
combine_horizontal("results/graphs/iv_curves_irradiance.png", 
                   "results/graphs/pv_curves_irradiance.png", 
                   "results/diagrams/composite_pv_irradiance.png")

combine_horizontal("results/graphs/iv_curves_temperature.png", 
                   "results/graphs/pv_curves_temperature.png", 
                   "results/diagrams/composite_pv_temperature.png")

# 2. RL Training curves (Reward & Tracking Efficiency)
combine_horizontal("results/graphs/phase_6/episode_reward_vs_episodes.png", 
                   "results/graphs/phase_6/tracking_efficiency_evaluation.png", 
                   "results/diagrams/composite_rl_training.png")

# 3. RL Losses (Critic & Actor)
combine_horizontal("results/graphs/phase_6/critic_loss_vs_steps.png", 
                   "results/graphs/phase_6/actor_loss_vs_steps.png", 
                   "results/diagrams/composite_rl_losses.png")

# 4. STC Power & Voltage Comparisons
combine_horizontal("results/graphs/phase_7/fig01_stc_pv_power_comparison.png", 
                   "results/graphs/phase_7/fig02_stc_pv_voltage_comparison.png", 
                   "results/diagrams/composite_stc_comparison.png")

# 5. Dynamic Transitions (Step & Combined Ramp)
combine_horizontal("results/graphs/phase_7/fig04_irradiance_step_response.png", 
                   "results/graphs/phase_7/fig06_combined_environmental_response.png", 
                   "results/diagrams/composite_dynamic_transitions.png")

# 6. Aggregate Comparison (Efficiency & Energy)
combine_horizontal("results/graphs/phase_7/fig07_tracking_efficiency_comparison.png", 
                   "results/graphs/phase_7/fig08_energy_harvested_comparison.png", 
                   "results/diagrams/composite_benchmark_metrics.png")

# 7. Statistical (Scenario-wise & Paired Difference)
combine_horizontal("results/graphs/phase_7/fig12_scenariowise_tracking_efficiency.png", 
                   "results/graphs/phase_7/fig13_paired_performance_difference_distribution.png", 
                   "results/diagrams/composite_statistical_eval.png")

print("All composite graphics generated successfully.")

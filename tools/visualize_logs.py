from omegaconf import DictConfig
import hydra

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_trajectories_log, split_logs_by_task, plot_trial_stats, alfworld_results_per_env_name_log, get_webshop_mean_score, get_webshop_mean_scores, mode_results

@hydra.main(version_base=None, config_path="../configs", config_name="visualize_logs")
def main(cfg: DictConfig) -> None:
    out = load_trajectories_log(path=cfg.log_path, run_name=cfg.run_name, load_dict=False)
    log = out['log']

    parsed_result = split_logs_by_task(text=log, num_tasks=cfg.num_tasks)
    assert len(parsed_result) == cfg.num_tasks

    res = plot_trial_stats(parsed_result=parsed_result, benchmark=cfg.benchmark.name, max_trials=4, save_path=f"{cfg.log_path}/{cfg.run_name}_logs_stats.png" if cfg.save_fig else None)
    if 'eval' in cfg.run_name:
        if cfg.benchmark.name == 'alfworld':
            print(alfworld_results_per_env_name_log(log, cfg.num_tasks, 1))
        elif cfg.benchmark.name == 'webshop':
            print(get_webshop_mean_score(log, cfg.num_tasks, 1))
        res = {k: v[-1] for k, v in res.items()}
    else:
        if cfg.benchmark.name == 'alfworld':
            print(alfworld_results_per_env_name_log(log, cfg.num_tasks, cfg.agent.max_reflection_depth+1))
        elif cfg.benchmark.name == 'webshop':
            print(get_webshop_mean_scores(log, cfg.num_tasks, cfg.agent.max_reflection_depth+1))

    print(res)

    ############################################
    ###         MODE OPTIONS SELECTION       ###
    ############################################
    # [react, reflection]
    # _[sum, mean, list]
    # _[token, count]
    # _[thought, action, observation, invalid]
    # _[traj, step]
    ############################################
    modes = [
        'react_mean_count_of_thought_per_traj', 
        'react_mean_count_of_action_per_traj', 
        'react_mean_count_of_observation_per_traj', 
        'react_mean_tokens_per_traj', 
        'react_mean_count_of_invalid_per_traj',
        ]
    if 'eval' not in cfg.run_name:
        modes += [
            'reflection1_mean_tokens_per_traj', 
            'reflection2_mean_tokens_per_traj', 
            'reflection3_mean_tokens_per_traj',
            'reflection1_mean_count_of_invalid_per_traj',
            'reflection2_mean_count_of_invalid_per_traj',
            'reflection3_mean_count_of_invalid_per_traj',]
    for mode in modes:
        print(f"{mode}: {mode_results(cfg.benchmark.name, log, cfg.num_tasks, mode)}")


if __name__ == '__main__':
    main()
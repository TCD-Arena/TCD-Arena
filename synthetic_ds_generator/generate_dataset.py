import hydra
from omegaconf import DictConfig
import os


import numpy as np
import pickle
import datetime
from omegaconf import OmegaConf




# Short script to generate a full ds according to config.
def generate(cfg: DictConfig) -> None:
    
    
    #seed numpy rng.
    gen = hydra.utils.instantiate( # nl samples defaults to 42. Change if necessary.
        cfg.generator,
        rng=cfg.seed,
        obs_n={"rng": cfg.seed},
        inno_n={"rng": cfg.seed},
        exog={"rng": cfg.seed},
        lagged={"rng": cfg.seed},
        instant={"rng": cfg.seed},
    )

    data_stack = []
    causal_stack = []
    exog_ts_stack = []
    exog_links_stack = []
    nl_stack = []
    nl_exogs_stack = []
    instant_stack = []
    nl_instant_stack = []
    resample_statistics = []

    f_name = cfg.output_dir + cfg.name + "/"
    if not os.path.exists(f_name):
        try:
            os.makedirs(f_name)
        except FileExistsError:
            print("Directory already exists")

    counter = 0
    error_counter = 0 
    while counter < cfg.n_samples:
        print(counter + 1, "/", cfg.n_samples)
        if cfg.link_mask_path:
            link_mask = np.load(cfg.link_mask_path + "/" + str(counter) + ".npy")
        if cfg.nl_mask_path:
            nl_mask = np.load(cfg.nl_mask_path + "/" + str(counter) + ".npy")
        if cfg.instant_link_mask_path:
            instant_mask = np.load(cfg.instant_link_mask_path + "/" + str(counter) + ".npy")
        if cfg.instant_nl_mask_path:
            instant_nl_mask = np.load(cfg.instant_nl_mask_path + "/" + str(counter) + ".npy")

        # Generate a sample. As the generation can fail occasionally based on e.g. faulty masks or problematic configuration we catch and retry here.
        # Note, the generator also maintains a number of internal error handling mechanisms on top of this.
        X, Y, Y_t0, exog_links, exog_ts, nl, nl_instant, nl_exogs, resamples = gen.get_sample(
            link_mask if cfg.link_mask_path else None,
            nl_mask if cfg.nl_mask_path else None,
            instant_mask if cfg.instant_link_mask_path else None,
            instant_nl_mask if cfg.instant_nl_mask_path else None,
        )
        
        # We additionally perform some tests here to ensure that non of the data samples contain NaNs or infs.
        c1 = np.isnan(X).sum() == 0
        c2 = np.isnan(Y).sum() == 0
        c3 = np.isnan(Y_t0).sum() == 0
        c4 = np.isinf(X).sum() == 0
        c5 = np.isinf(Y).sum() == 0
        c6 = np.isinf(Y_t0).sum() == 0
        c7 = X.var(axis=1).min() > 0.01
        c8 = np.isnan(exog_ts).sum() == 0 if isinstance(exog_ts, np.ndarray) else True
        c9 = np.isinf(exog_ts).sum() == 0 if isinstance(exog_ts, np.ndarray) else True
        c10 = np.isnan(exog_links).sum() == 0 if isinstance(exog_links, np.ndarray) else True
        c11 = np.isinf(exog_links).sum() == 0 if isinstance(exog_links, np.ndarray) else True

        if (c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9 and c10 and c11):
          
            data_stack.append(X)
            causal_stack.append(Y)
            instant_stack.append(Y_t0)
            exog_ts_stack.append(exog_ts)
            exog_links_stack.append(exog_links)
            nl_stack.append(nl)
            nl_exogs_stack.append(nl_exogs)
            nl_instant_stack.append(nl_instant)
            resample_statistics.append([error_counter] + resamples)
    
            counter += 1
        else:
            print("Samples did not pass tests. Retrying...")
            print((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))
            error_counter+=1

    if cfg.run_name is not None:
        output_dir = os.path.join(f_name, str(cfg.run_name)+ datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f")) 
    else:
        # If we exexute without multirun there is no job_id which we use for naming
        output_dir = os.path.join(f_name, "0" + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}")

    def save_array(filename, array):
        np.save(os.path.join(output_dir, filename), np.array(array))

    save_array("X.npy", data_stack)
    save_array("Y.npy", causal_stack)
    save_array("instant_links.npy", instant_stack)
    print("Resample statistics (number of failed attempts processes, number of failed sections):")
    print(resample_statistics)
    save_array("resample_statistics.npy", resample_statistics)


    if nl_stack and isinstance(nl_stack[0], np.ndarray):
        save_array("nl.npy", nl_stack)
    if nl_instant_stack and isinstance(nl_instant_stack[0], np.ndarray):
        save_array("nl_instant.npy", nl_instant_stack)
    if nl_exogs_stack and isinstance(nl_exogs_stack[0], np.ndarray):
        save_array("nl_exogs.npy", nl_exogs_stack)
    if exog_ts_stack and isinstance(exog_ts_stack[0], np.ndarray):
        save_array("exog_ts.npy", exog_ts_stack)
    if exog_links_stack and isinstance(exog_links_stack[0], np.ndarray):
        save_array("exog_links.npy", exog_links_stack)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)


@hydra.main(version_base="1.3", config_path="config", config_name="generate_dataset.yaml")
def main(cfg: DictConfig):
    generate(cfg)


if __name__ == "__main__":
    main()

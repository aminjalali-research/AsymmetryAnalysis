# What's Done                                                                                                                                               
```                                                                                                                                                            
  - 6 TUH corpora downloaded and verified                                                                                                                   
  - Subject registry built (3,124 subjects, cross-corpus overlap mapped)                                                                                    
  - Subject-disjoint splits created (stratified, all datasets in every split)
  - Dataset loaders updated (10s windows, unified splits, new TUSL loader)
  - Configs updated (6 tasks, temporal context=3, 10s epochs)
  - Paper updated
  - All 180 tests passing

  What's Next (in order)

  1. Regenerate preprocessed caches (10s windows)

  The old 5s caches were deleted. Before training can start, we need to preprocess the raw EDFs into cached .pt tensors with the new 10s window size. This
  is a one-time CPU job that takes a few hours but saves enormous time during training.

  2. Regenerate instruction data

  The current data/eeg_instruct/ (36,820 samples) was generated with 5s epochs and only 4 tasks. It needs regeneration to cover all 6 tasks, 10s windows,
  and use the unified train split subjects only.

  3. Re-run Stage 1 (Foundation Fine-tuning)

  Now with 6 tasks (+ slowing + epilepsy) across all 6 corpora, with the subject-disjoint splits. This is the big training run — multi-task with Kendall
  uncertainty weighting.

  4. Re-run Stages 2–4

  - Stage 2: EEG-language alignment (same architecture, new data)
  - Stage 3: Instruction tuning (on regenerated instruction data)
  - Stage 4: GRPO RL (with all datasets in eval)

  5. Run ablations on new data

  All ablation scripts (D, F, G) need re-running with the new splits/windows so results are apples-to-apples.

  6. Evaluate on ALL datasets

  Final evaluation covers all 6 corpora across the 5 clinical workflow phases — this is the key change from before where we only tested on TUSZ.

  My recommendation: Start with step 1 (preprocessing cache) since everything else depends on it. Want me to build the preprocessing script for the new 10s
  windows?

Suggested Workflow                                                                                                                                        
   
  Phase 1: Local Validation (your RTX 5070 Ti)                                                                                                              
                                       
  Goal: Confirm everything runs end-to-end with small data before burning cluster hours.                                                                    
                                       
  1. Smoke test each dataset loader — Load 2-3 recordings from each corpus, verify shapes, labels, and split filtering work                                 
  2. Mini Stage 1 run — Train for 2-3 epochs with max_recordings=10 per dataset, confirm all 6 task heads produce valid losses
  3. Mini Stages 2-4 — Same idea, just verify the pipeline doesn't crash                                                                                    
  4. Fix any bugs locally — Much faster iteration than on a remote cluster                                                                                  
                                                                                                                                                            
  Phase 2: Cluster Preparation                                                                                                                              
                                                                                                                                                            
  What you need for CCDB/Alliance:                                                                                                                          
                                               
  1. Job scripts (SLURM .sh files) for each stage — they use sbatch with #SBATCH directives                                                                 
  2. Environment setup — Either a requirements.txt install into a virtualenv, or a conda env export
  3. Data transfer — rsync your data/ directory to the cluster's scratch space (or /project/)                                                               
  4. Multi-GPU config — Your code uses DeepSpeed already (requirements.txt has it). Need to configure for their node topology (typically 4× A100 or 4× V100 
  per node)                                                                                                                                                 
                                                                                                                                                            
  Phase 3: Cluster Execution                                                                                                                                
                                               
  Stage 1 (Foundation)     → 1 node, 4 GPUs, ~12-24 hrs                                                                                                     
  Stage 2 (Alignment)      → 1 node, 4 GPUs, ~6-12 hrs                                                                                                      
  Stage 3 (Instruction)    → 1 node, 1-2 GPUs, ~2-4 hrs
  Stage 4 (RL)             → 1 node, 1-2 GPUs, ~4-8 hrs                                                                                                     
  Ablations (D, F, G)      → Can run in parallel across nodes                                                                                               
                                                                                                                                                            
  What I'd build for you:                                                                                                                                   
                                                                                                                                                            
  1. scripts/smoke_test_pipeline.py — Quick local validation that loads every dataset, runs a mini forward pass through the encoder, and verifies all 6 task
   losses                                      
  2. cluster/ directory with:                                                                                                                               
    - setup_env.sh — Creates conda/venv on Alliance cluster
    - stage1.sh through stage4.sh — SLURM job scripts                                                                                                       
    - transfer_data.sh — rsync to cluster scratch                                                                                                           
    - run_all.sh — Orchestrates the full pipeline                                                                                                           
                                                                                                                                                            
  Do you know which Alliance cluster you'll use (Narval, Cedar, Graham, Niagara)? And how many GPUs you want per job? That affects the SLURM configuration. 
                                                       
                                                                                                    
For your use case, here's the breakdown:                                                                                                     
                                                                                                                                       
    Use /project/def-amiilab/aminj/ (where we're currently transferring) for:                                                                    
    - Pretrained weights (~7 GB) — persistent, shared across jobs
    - Code, configs, split files — persistent                                                                                                    
    - Raw datasets (~200 GB) — persistent, don't want to re-transfer
                                               
    This is the right choice because:
    - You have 10 TB project storage quota (from your allocation)
    - Data persists indefinitely (not purged)
    - Shared across all cluster nodes

    Use /scratch/aminj/ for:
    - Training outputs (checkpoints, logs, intermediate results)
    - Preprocessed caches (can be regenerated)
    - Anything temporary or reproducible

    Don't use /home/aminj/ — only 50 GB quota, too small.

    Bottom line: What we're doing now (transferring to /project/) is correct. The 200 GB of raw datasets should live in project space since you
    don't want to re-transfer them. Just put training outputs in /scratch/ to avoid filling up your project quota. You could update the SLURM
    scripts to write checkpoints to /scratch/aminj/ReportGen/checkpoints/ instead.



what is the environment you created? conda or something different?                                                                        
                                       
    On Rorqual I created a virtualenv (Python's built-in virtualenv), not conda.                                                                 
                                                                                                                                                 
    This is because Alliance Canada clusters don't support conda — they use a module system (module load python/3.10) combined with virtualenv   
    and pre-built wheels (pip install --no-index torch). Their docs explicitly recommend this approach.                                          
                                                                                                                                                 
    The environment is at:                                                                                                                       
    /project/def-amiilab/aminj/envs/reportgen/                                                                                                   
                                                                                                                                                 
    Activated with:                                                                                                                              
    module load python/3.10 cuda/12.2 cudnn                                                                                                      
    source /project/def-amiilab/aminj/envs/reportgen/bin/activate

    This is handled automatically by cluster/common.sh which all SLURM job scripts source.

    Your local machine still uses conda (conda activate reportgen), so the two environments are independent but have the same packages.
```


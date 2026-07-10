# Zero-Shot Coordination between Teams of Agents: N-XPlay
This is the repository for **Towards Zero-Shot Coordination between Teams of Agents: The N-XPlay Framework** paper presented at RSS Workshop on Scalable and Resilient Multi-Robot Systems: Decision-Making, Coordination, and Learning 2025. 

Paper link: https://arxiv.org/abs/2506.17560. Please email the authors at first.lastname@colorado.edu if you have any questions!

## Set up guide
1. Follow setup instrcution for overcooked_ai from [here](https://github.com/HIRO-group/overcooked_ai)
2. Clone this repository: `git clone git@github.com:HIRO-group/multiHRI.git`
3. Activate conda env: `conda activate mHRI`
4. Run: `pip install pip==24.0 wheel==0.38.4 setuptools==65.5.0`
4. cd into the repo and run: `pip install -e .`.
5. Install liblsl using conda 

## Training agents
Refer to scripts/train_agents.py for examples on how to train different agent trainings.
If you've installed the package as above, you can run the script using:  
`python scripts/train_agents.py`

## Citation
If you use this repository in any way, please cite:
```
{@inproceedings{abderezaei2025zeroshotcoordinationteamsagents,
    author      = {Abderezaei, Ava and Lin, Chi-Hui and Miceli, Joseph and Sivagnanadasan, Naren and Aroca-Ouellette, Stéphane and Brawer, Jake and Roncone, Alessandro},
    title       = {Towards Zero-Shot Coordination between Teams of Agents: The N-XPlay Framework},
    booktitle   = {RSS Workshop on Scalable and Resilient Multi-Robot Systems: Decision-Making, Coordination, and Learning},
    year        = {2025},
    organization= {RSS},
    month       = {June},
    numpages    = {3},
}
```

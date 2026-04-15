# DreamerV3 Learning Resources

A curated reading / watching list for people coming into this repo. The
goal is to give you enough background to read the DreamerV3 paper
comfortably, then enough follow-ups to go deeper.

Links marked [paper] are arXiv PDFs. Links marked [code] are official
repos. Links marked [talk] are recorded presentations.

---

## Start here

1. **DreamerV3 paper** — Hafner, Pasukonis, Ba, Lillicrap. *Mastering
   Diverse Domains through World Models.* 2023.
   - [paper] <https://arxiv.org/abs/2301.04104>
   - [code] <https://github.com/danijar/dreamerv3>
   - [project page] <https://danijar.com/project/dreamerv3/>

2. **Danijar Hafner's homepage** — the author's publications and talks,
   including earlier Dreamer iterations and related work.
   - <https://danijar.com/>

---

## The Dreamer lineage

Read these in order to see how the architecture evolved:

1. **PlaNet** — Hafner et al. *Learning Latent Dynamics for Planning from
   Pixels.* ICML 2019. Introduces the RSSM used by all later Dreamers.
   - [paper] <https://arxiv.org/abs/1811.04551>
   - [code] <https://github.com/google-research/planet>

2. **Dreamer (V1)** — Hafner et al. *Dream to Control: Learning Behaviors
   by Latent Imagination.* ICLR 2020. First version of imagined
   actor-critic training on top of the RSSM.
   - [paper] <https://arxiv.org/abs/1912.01603>
   - [code] <https://github.com/danijar/dreamer>

3. **DreamerV2** — Hafner et al. *Mastering Atari with Discrete World
   Models.* ICLR 2021. Switches to categorical latents and beats Atari.
   - [paper] <https://arxiv.org/abs/2010.02193>
   - [code] <https://github.com/danijar/dreamerv2>

4. **DreamerV3** — see above. Adds symlog predictions, two-hot targets,
   and a single fixed set of hyperparameters that works across all domains
   the authors tried.

---

## Background you may want first

If the jargon in the paper is unfamiliar, these are good primers:

- **World models in general.** Ha & Schmidhuber, *World Models.* 2018.
  [paper] <https://arxiv.org/abs/1803.10122>. Also has an excellent
  interactive write-up at <https://worldmodels.github.io/>.
- **Model-based RL survey.** Moerland et al., *Model-based Reinforcement
  Learning: A Survey.* 2022. [paper] <https://arxiv.org/abs/2006.16712>
- **Variational autoencoders.** Kingma & Welling, *Auto-Encoding
  Variational Bayes.* 2013. [paper] <https://arxiv.org/abs/1312.6114>.
  The KL between RSSM prior and posterior is the same idea.
- **Actor-critic methods.** Sutton & Barto, *Reinforcement Learning: An
  Introduction*, 2nd ed., Chapter 13. Free PDF:
  <http://incompleteideas.net/book/the-book-2nd.html>.

---

## Benchmarks DreamerV3 is evaluated on

- **DeepMind Control Suite.** Tassa et al., 2018.
  [paper] <https://arxiv.org/abs/1801.00690>
  [code] <https://github.com/google-deepmind/dm_control>
- **Arcade Learning Environment (Atari).** Bellemare et al., 2013.
  [paper] <https://arxiv.org/abs/1207.4708>
  [code] <https://github.com/Farama-Foundation/Arcade-Learning-Environment>
- **Crafter.** Hafner, 2021 — a fast 2D Minecraft-inspired benchmark with
  22 achievements. DreamerV3's flagship result before Minecraft proper.
  [paper] <https://arxiv.org/abs/2109.06780>
  [code] <https://github.com/danijar/crafter>
- **MineRL / Minecraft.** Guss et al., 2019 — the Minecraft environment
  DreamerV3 uses for its diamond-collection result.
  [paper] <https://arxiv.org/abs/1907.13440>
  [code] <https://github.com/minerllabs/minerl>
  [docs] <https://minerl.readthedocs.io/>
- **MiniGrid.** Chevalier-Boisvert et al., 2018 — a lightweight grid-world
  benchmark suite.
  [code] <https://github.com/Farama-Foundation/Minigrid>
- **Gymnasium.** Successor to OpenAI Gym, the API every environment above
  exposes.
  [docs] <https://gymnasium.farama.org/>

---

## Talks and videos

These are recorded presentations by the DreamerV3 authors and the broader
community. All links are to the official channel of the speaker or
conference — follow them to find the specific talks.

- **Danijar Hafner's YouTube channel.** Contains talks on PlaNet,
  Dreamer, DreamerV2, DreamerV3, and project trailer videos of trained
  agents on DMC, Atari, Crafter, and Minecraft.
  <https://www.youtube.com/@danijarhafner>
- **DeepMind YouTube channel** — DreamerV3 and related world-model work
  appear in DeepMind's research talks.
  <https://www.youtube.com/@Google_DeepMind>
- **Two Minute Papers** — accessible 5-minute overview videos; search the
  channel for "DreamerV3". <https://www.youtube.com/@TwoMinutePapers>
- **Yannic Kilcher** — in-depth paper walk-through videos. Search the
  channel for "DreamerV3". <https://www.youtube.com/@YannicKilcher>

If you want video of trained agents specifically, the project page
(<https://danijar.com/project/dreamerv3/>) hosts the official rollouts.
You can also generate your own using `scripts/record.py` in this repo —
see the [README](../README.md#recording-rollout-videos).

---

## Blog posts and secondary write-ups

- **Lil'Log** by Lilian Weng — several posts on world models,
  model-based RL, and policy gradient methods that pair well with
  Dreamer. <https://lilianweng.github.io/>
- **The Gradient** and **Distill** archives are good non-paper sources
  on RL fundamentals. <https://thegradient.pub/> ·
  <https://distill.pub/>

---

## Going further

Areas where DreamerV3 has inspired follow-up research — good rabbit
holes once you've internalized the base method:

- **Offline world models.** Training Dreamer-style agents from fixed
  datasets without any environment interaction.
- **Transformer world models.** Replacing the GRU with a Transformer —
  see e.g. *TransDreamer* and *IRIS*.
- **Scaling laws for world models.** How capacity and data interact for
  model-based RL.
- **Exploration via world-model uncertainty.** Using disagreement across
  imagined rollouts as an intrinsic reward.

For all of these, the [project page](https://danijar.com/project/dreamerv3/)
and the citations in the DreamerV3 paper are the best next step.

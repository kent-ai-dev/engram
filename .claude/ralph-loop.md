# Engram Autonomous Loop — Natural Progression v15→v16→v18

## Operating Principle
Continue v15 branches, then v16 FUTURE_RESEARCH candidates. Only escalate to v18 substrate shift if it becomes clear we need a different strategy. Do not ask for permission. **MANDATORY after every training iteration: (1) update GitHub Pages training blog, (2) push latest model to Modal, (3) commit to git.**

## Phase 1: v15 Branches (escalation from confirmed v14-B vocab bottleneck)
Currently training: v15-A (stronger adaptive pondering, cap 3→5, cost 0.05→0.02)
Queue: v15-B (episodic memory in training), v15-C (surprise modulation), v15-D (other levers)

For each v15 branch:
1. Wait for training to complete
2. Evaluate on 5-prompt chitchat benchmark
3. Check for ENGRAM_COHERENT signal or multi-turn coherence
4. **MANDATORY POST-ITERATION:**
   a. Update GitHub Pages training blog (engram-page-updater: both kent-ai-dev.github.io/engram/ and claw-journal/engram.html)
   b. Push trained model weights to Modal (ensure latest version available for inference + analysis)
   c. Commit training results and metadata to git
5. Log outcome, proceed to next v15 branch

## Phase 2: v16 Candidates (if v15 all complete without ENGRAM_COHERENT)
From FUTURE_RESEARCH.md, in order of investment:
1. ∇-Reasoner (test-time gradient descent, $0 training, prototypable against any model)
2. Recurrent depth unbounded (if Branch A shows pondering is engaged, $10-15)
3. TTT for episodic memory (if Branch B shows episodic works, $10-15)
4. PCN (if Branch D/C show instability, $15-20)

After each v16 run: MANDATORY — update blogs + push to Modal + commit results

## Phase 3: v18 Substrate Shift (only if clear we need different strategy)
From V18_STALK_VOCAB.md — test whether meaning-as-distribution beats meaning-as-point
- Build eval_polysemy.py (30 homonym OOD benchmark)
- Run v18-control, v18-A, v18-B, v18-C
Pass condition: v18-B beats v18-control by ≥3 points + no chitchat regression + non-degenerate Λ

## Continuous Operation Rules
- No permission prompts — investigate and fix blockers autonomously
- Retry failed runs once with adjusted LR before escalating
- **MANDATORY: After every training iteration, update both GitHub Pages blogs AND push to Modal AND commit to git (non-negotiable)**
- Track cumulative spend against $150 ceiling
- Loop indefinitely until ENGRAM_COHERENT or explicit stop signal

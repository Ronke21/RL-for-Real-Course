# Helpers shared by lanes.sh. Expects STEP_GPU and STEP_COURSE exported by the caller.
C="/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students/claude_run"
W="$C/work"
WD="$C/work_dr"
say() { echo "$(date +%T) [gpu$STEP_GPU] $*"; }
cw() { echo "$C/${STEP_COURSE:-work}"; }       # current course copy
step() { local name=$1; say "start $name"; bash "$C/step.sh" "$@"; local rc=$?; say "end $name rc=$rc"; return $rc; }
evalp() {  # evalp NAME OUTPUT args...  (physical_eval, 500 episodes, 128 envs); skips if report exists
  local name=$1 out=$2; shift 2
  [ -s "$(cw)/$out" ] && { say "skip $name ($out exists)"; return 0; }
  step "$name" 20 python -m course.physical_eval --num-envs 128 --episodes 500 --output "$out" "$@"
}
train() {  # train NAME MINUTES MODULE args...   retries with fewer envs on CUDA OOM; ok if last.ckpt exists
  local name=$1 min=$2 mod=$3; shift 3
  [ -s "$(cw)/results/$name/num_envs.txt" ] && { say "skip train $name (done)"; return 0; }
  for n in 4096 2048 1024; do
    rm -rf "$(cw)/results/$name"
    step "$name" $((min + 5)) python -m course.launch "$mod" train --experiment-name "$name" --headless --seed 0 \
         --num-envs "$n" --batch-size "$n" "$@"
    local rc=$?
    if [ -s "$(cw)/results/$name/last.ckpt" ] && { [ $rc -eq 0 ] || [ $rc -eq 124 ]; }; then echo "$n" > "$(cw)/results/$name/num_envs.txt"; return 0; fi
    tail -300 "$C/logs/$name.log" | grep -qi "out of memory" || return $rc
    say "$name OOM at $n envs; retrying smaller"
  done
  return 1
}
m3_start() {  # my M1 policy if >= 80% full-clip survival, else the supplied expert (guide's fallback)
  python3 -c "
import json
try:
    d=json.load(open('$W/results/m1_control.json')); s=d.get('summary',d)['survival_success_rate']
    print('results/m1_control/last.ckpt' if s>=0.8 else 'course/pretrained/m1/last.ckpt')
except Exception: print('course/pretrained/m1/last.ckpt')"
}

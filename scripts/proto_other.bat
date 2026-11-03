@echo off
cd /d "%~dp0.."
rem On-device prototype adaptation + explicit "other sign" class (about 1-1.5 h).
rem Finished steps are skipped, so after a power cut just run this file again.
call conda activate tinyml 2>nul
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
if not exist proto_adapt.txt (
  echo 1/3 prototype adaptation, no back-propagation ...
  python src\proto_adapt.py --data ts1m_ma_32_clean --morocco morocco_crops --runs runs\c32_mild40k_s42 runs\c32_mild40k_s1 runs\c32_mild40k_s2 > runs\proto_adapt.log 2>&1
)
if not exist ts1m_other_32\summary.txt (
  echo 2/3 building the other-sign set from the TS-1M zip ...
  python src\make_other_set.py --zip "ts-1m-datset.zip" --out ts1m_other_32 > runs\make_other.log 2>&1
)
for %%S in (42 1 2) do (
  if not exist runs\c32_mild40k_s%%S_other\other_results.txt (
    echo 3/3 seed %%S: training with an explicit other-sign class ...
    python src\train_other.py --run runs\c32_mild40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --other ts1m_other_32 --out runs\c32_mild40k_s%%S_other --seed %%S > runs\c32_mild40k_s%%S_other.log 2>&1
  )
)
echo done. Results: proto_adapt.txt and runs\c32_mild40k_s*_other\other_results.txt

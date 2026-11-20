set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONDONTWRITEBYTECODE=1
analysis=method_validation/analysis

case "${1:-}" in
    "")
        python3 "$analysis/verify_integrity.py"
        ;;
    --with-model)
        python3 method_validation/data/build_test_data.py
        python3 "$analysis/run_pilot_annotations.py" --workers "${MV_PILOT_WORKERS:-6}"
        python3 "$analysis/run_annotations.py" --workers "${MV_ANNOTATION_WORKERS:-32}"
        python3 "$analysis/run_shared_llm.py" --workers "${MV_LLM_WORKERS:-8}"
        python3 "$analysis/run_methods.py"
        python3 "$analysis/run_path_ratings.py" --workers "${MV_PATH_WORKERS:-12}"
        ;;
    *)
        printf 'Usage: bash %s [--with-model]\n' "$analysis/reproduce.sh" >&2
        exit 2
        ;;
esac

python3 "$analysis/analyze_results.py"
python3 "$analysis/verify_integrity.py"
python3 -m unittest discover -s tests -v

import os, json, pandas as pd, numpy as np, xgboost as xgb
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    CUDA_AVAILABLE = False
from datetime import date
import sys
from docx import Document
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.join(BASE_DIR, 'JD', 'job_description.docx')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resume_filter.json')
INPUT_PATH = os.path.join(BASE_DIR, 'candidates', 'candidates.jsonl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'submission.csv')

EPOCH_BASE_ORDINAL = date(1970, 1, 1).toordinal()


def parse_date_to_days(value):
    if not value:
        return 0
    try:
        return date.fromisoformat(value).toordinal() - EPOCH_BASE_ORDINAL
    except ValueError:
        return 0


def preferred_work_mode_one_hot(value):
    return {
        'preferred_work_mode_remote': 1 if value == 'remote' else 0,
        'preferred_work_mode_hybrid': 1 if value == 'hybrid' else 0,
        'preferred_work_mode_onsite': 1 if value == 'onsite' else 0,
        'preferred_work_mode_flexible': 1 if value == 'flexible' else 0,
    }


def build_feature_row(c, ssum, scar, ssk):
    r = c.get('redrob_signals', {})
    skill_assess = r.get('skill_assessment_scores', {})
    skill_avg = np.mean(list(skill_assess.values())) if len(skill_assess) > 0 else 0.0
    skill_count = len(skill_assess)
    salary_range = r.get('expected_salary_range_inr_lpa', {})
    salary_min = float(salary_range.get('min', 0))
    salary_max = float(salary_range.get('max', 0))
    work_mode = r.get('preferred_work_mode', '')
    work_mode_features = preferred_work_mode_one_hot(work_mode)

    row = {
        'sim_summary': float(ssum),
        'sim_career': float(scar),
        'sim_skills': float(ssk),
        'years_experience': float(c.get('profile', {}).get('years_of_experience', 0)),
        'profile_completeness_score': float(r.get('profile_completeness_score', 0)),
        'github_activity_score': float(r.get('github_activity_score', -1)),
        'connection_count': int(r.get('connection_count', 0)),
        'applications_submitted_30d': int(r.get('applications_submitted_30d', 0)),
        'open_to_work_flag': 1 if r.get('open_to_work_flag', False) else 0,
        'avg_response_time_hours': float(r.get('avg_response_time_hours', 0)),
        'profile_views_received_30d': int(r.get('profile_views_received_30d', 0)),
        'endorsements_received': int(r.get('endorsements_received', 0)),
        'notice_period_days': int(r.get('notice_period_days', 0)),
        'expected_salary_min': salary_min,
        'expected_salary_max': salary_max,
        'salary_range_width': salary_max - salary_min,
        'skill_assess_avg': float(skill_avg),
        'skill_assess_count': int(skill_count),
        'search_appearance_30d': int(r.get('search_appearance_30d', 0)),
        'saved_by_recruiters_30d': int(r.get('saved_by_recruiters_30d', 0)),
        'interview_completion_rate': float(r.get('interview_completion_rate', 0)),
        'offer_acceptance_rate': float(r.get('offer_acceptance_rate', -1)),
        'verified_email': 1 if r.get('verified_email', False) else 0,
        'verified_phone': 1 if r.get('verified_phone', False) else 0,
        'linkedin_connected': 1 if r.get('linkedin_connected', False) else 0,
        'willing_to_relocate': 1 if r.get('willing_to_relocate', False) else 0,
        'signup_day': parse_date_to_days(r.get('signup_date', '')),
        'last_active_day': parse_date_to_days(r.get('last_active_date', '')),
    }
    row.update(work_mode_features)
    return row


# AI-skill anchors and threshold (match training logic)
AI_SKILL_ANCHORS = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'natural language processing', 'computer vision', 'generative AI',
    'large language model', 'neural network', 'data science',
    'model deployment', 'model training', 'prompt engineering'
]
AI_SKILL_SIM_THRESHOLD = 0.37  


def count_ai_skills_semantic(skills, anchor_embs, encoder, threshold=AI_SKILL_SIM_THRESHOLD):
    if not skills:
        return 0
    names = [s.get('name', '').strip() for s in skills if isinstance(s, dict) and s.get('name')]
    if not names:
        return 0
    from sentence_transformers import util
    skill_embs = encoder.encode(names, convert_to_tensor=True, batch_size=64)
    sim_scores = util.cos_sim(skill_embs, anchor_embs).cpu().numpy()
    max_scores = np.max(sim_scores, axis=1)
    return int((max_scores >= threshold).sum())


def generate_submission():
    model = xgb.Booster(); model.load_model(MODEL_PATH)
    # Always compute features at runtime (do not rely on a potentially stale candidate_features.csv)
    feats_path = os.path.join(BASE_DIR, 'models', 'feature_metadata.json')
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Feature metadata not found at {feats_path}. Run train_model first.")
    with open(feats_path, 'r') as fh:
        meta = json.load(fh)
    feature_list = meta.get('features', [])

    # Load JSONL file (each line is a separate JSON object)
    data = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    results = []

    # Require sentence-transformers and torch to compute embeddings at runtime
    if torch is None:
        print("Torch is unavailable. Install torch or run train_model.py to precompute features.")
        sys.exit(1)
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        print("sentence-transformers (or its dependencies) failed to import. Please install sentence-transformers and ensure torch works, or run train_model.py to precompute features.")
        sys.exit(1)

    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    # Enable multi-threading for CPU encoding efficiency
    if device == 'cpu':
        encoder.max_seq_length = 256  # Reduce for faster processing on CPU
    jd_text = " ".join([p.text for p in Document(JD_PATH).paragraphs])
    jd_emb = encoder.encode(jd_text, convert_to_tensor=True)
    ai_anchor_embs = encoder.encode(AI_SKILL_ANCHORS, convert_to_tensor=True)

    # BATCH PROCESSING: Extract all text fields first, then encode in batches
    print("Extracting candidate text fields...")
    summaries = []
    careers = []
    skills_texts = []
    candidate_info = []
    
    for c in tqdm(data, desc="Extracting text fields"):
        summary = str(c.get('profile', {}).get('summary', ''))
        career = " ".join([ch.get('description', '') for ch in c.get('career_history', [])])
        skills_text = " ".join([s.get('name', '') for s in c.get('skills', [])])
        
        summaries.append(summary)
        careers.append(career)
        skills_texts.append(skills_text)
        candidate_info.append(c)
    
    # Batch encode all summaries, careers, and skills at once (much faster)
    print("Batch encoding summaries...")
    sum_embs = encoder.encode(summaries, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    print("Batch encoding careers...")
    career_embs = encoder.encode(careers, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    print("Batch encoding skills...")
    skills_embs = encoder.encode(skills_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    
    # Compute all similarities at once
    print("Computing similarities...")
    sim_sums = util.cos_sim(jd_emb, sum_embs).cpu().numpy().flatten()
    sim_careers = util.cos_sim(jd_emb, career_embs).cpu().numpy().flatten()
    sim_skills_all = util.cos_sim(jd_emb, skills_embs).cpu().numpy().flatten()
    
    # Process results - Batch compute all features and predictions
    print("Computing AI skill counts...")
    ai_skill_counts = []
    for c in tqdm(candidate_info, desc="AI skill matching"):
        ai_skill_count = count_ai_skills_semantic(c.get('skills', []), ai_anchor_embs, encoder)
        ai_skill_counts.append(ai_skill_count)
    
    print("Building feature rows...")
    all_rows = []
    for idx, c in tqdm(enumerate(candidate_info), total=len(candidate_info), desc="Building features"):
        sim_sum = float(sim_sums[idx])
        sim_career = float(sim_careers[idx])
        sim_skills = float(sim_skills_all[idx])
        row = build_feature_row(c, sim_sum, sim_career, sim_skills)
        all_rows.append(row)
    
    # Batch predict on all candidates at once
    print("Running XGBoost predictions...")
    feats_df = pd.DataFrame([[row.get(k, 0) for k in feature_list] for row in all_rows], columns=feature_list)
    all_scores = model.predict(xgb.DMatrix(feats_df))
    
    print("Assembling results...")
    for idx, c in tqdm(enumerate(candidate_info), total=len(candidate_info), desc="Assembling results"):
        sim_sum = float(sim_sums[idx])
        sim_career = float(sim_careers[idx])
        sim_skills = float(sim_skills_all[idx])
        ai_skill_count = ai_skill_counts[idx]
        r = c.get('redrob_signals', {})
        resp_rate = float(r.get('recruiter_response_rate', 0.0))
        years = float(c.get('profile', {}).get('years_of_experience', 0.0))
        title = c.get('profile', {}).get('current_title', '')
        results.append({
            'candidate_id': c.get('candidate_id'),
            'ai_skill_count': ai_skill_count,
            'sim_summary': sim_sum,
            'sim_career': sim_career,
            'sim_skills': sim_skills,
            'response_rate': resp_rate,
            'reasoning': f"{title} with {years:.1f} yrs; {ai_skill_count} AI core skills; response rate {resp_rate:.2f}."
        })
    
    # 2. Direct Composite Score (JD fit + AI skills + engagement)
    # Rank by: AI skills (normalized) + semantic similarities + response rate
    # All candidates ranked on intrinsic JD fit, not predicted behavior.
    df = pd.DataFrame(results)
    
    # Normalize AI skill count (0-15 -> 0-1 scale, roughly)
    max_ai_skills = df['ai_skill_count'].max() if 'ai_skill_count' in df.columns else 1
    max_ai_skills = max(max_ai_skills, 1)  # avoid division by zero
    
    # Direct composite: equal weight to JD semantic fit and engagement signals
    df['score'] = (
        (df['ai_skill_count'] / max_ai_skills) * 0.20  # AI skill relevance
        + df['sim_summary'] * 0.20  # Summary/profile match
        + df['sim_career'] * 0.15  # Career history match
        + df['sim_skills'] * 0.25  # Skills match
        + df['response_rate'] * 0.20 # Engagement/responsiveness
    )
    
    # Rescale to 0.2-0.992 range for submission format
    min_s, max_s = df['score'].min(), df['score'].max()
    if max_s - min_s < 1e-6:
        df['score'] = 0.6
    else:
        df['score'] = 0.2 + (df['score'] - min_s) * (0.992 - 0.2) / (max_s - min_s)

    # Use rounded output score for ordering and tie-breaks to match CSV output
    df['score_out'] = df['score'].round(4)

    # 3. Finalize: sort by rounded score desc, tie-break candidate_id asc, take top 100, assign ranks
    df = df.sort_values(by=['score_out', 'candidate_id'], ascending=[False, True])
    df_top = df.head(100).copy()
    df_top['rank'] = range(1, len(df_top) + 1)

    # Write required header and UTF-8 encoding exactly using 4 decimal places
    out_df = df_top[['candidate_id', 'rank', 'score_out', 'reasoning']]
    out_df = out_df.rename(columns={'score_out': 'score'})
    out_df['score'] = out_df['score'].map(lambda v: f"{v:.4f}")
    out_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Comparative submission saved to {OUTPUT_PATH} ({len(out_df)} rows).")

if __name__ == "__main__":
    generate_submission()
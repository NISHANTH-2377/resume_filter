import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
import torch
from datetime import date
from sentence_transformers import SentenceTransformer, util
from docx import Document
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.join(BASE_DIR, 'JD', 'job_description.docx')
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'candidates.jsonl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resume_filter.json')

# Initialize sentence embedding model (uses GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = SentenceTransformer('all-mpnet-base-v2', device=device)

def get_jd_text(path):
    doc = Document(path)
    return " ".join([p.text for p in doc.paragraphs])

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


AI_SKILL_ANCHORS = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'natural language processing', 'computer vision', 'generative AI',
    'large language model', 'neural network', 'data science',
    'model deployment', 'model training', 'prompt engineering'
]
AI_SKILL_SIM_THRESHOLD = 0.35


def count_ai_skills_semantic(skills, anchor_embs, threshold=AI_SKILL_SIM_THRESHOLD):
    if not skills:
        return 0
    names = [s.get('name', '').strip() for s in skills if isinstance(s, dict) and s.get('name')]
    if not names:
        return 0
    skill_embs = encoder.encode(names, convert_to_tensor=True)
    sim_scores = util.cos_sim(skill_embs, anchor_embs).cpu().numpy()
    max_scores = np.max(sim_scores, axis=1)
    return int((max_scores >= threshold).sum())


def build_feature_row(c, ssum, scar, ssk, ai_skill_count):
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
        'ai_skill_count': int(ai_skill_count),
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


def train_model():
    jd_text = get_jd_text(JD_PATH)
    jd_emb = encoder.encode(jd_text, convert_to_tensor=True)
    ai_anchor_embs = encoder.encode(AI_SKILL_ANCHORS, convert_to_tensor=True)
    booster = None
    chunk_size = 5000 
    
    print("Training model with dynamic AI skill matching...")
    
    feature_metadata = {
        'features': [
            'sim_summary', 'sim_career', 'sim_skills', 'ai_skill_count',
            'years_experience', 'profile_completeness_score', 'github_activity_score',
            'connection_count', 'applications_submitted_30d', 'open_to_work_flag',
            'avg_response_time_hours', 'profile_views_received_30d', 'endorsements_received',
            'notice_period_days', 'expected_salary_min', 'expected_salary_max', 'salary_range_width',
            'skill_assess_avg', 'skill_assess_count', 'search_appearance_30d',
            'saved_by_recruiters_30d', 'interview_completion_rate', 'offer_acceptance_rate',
            'verified_email', 'verified_phone', 'linkedin_connected', 'willing_to_relocate',
            'signup_day', 'last_active_day',
            'preferred_work_mode_remote', 'preferred_work_mode_hybrid',
            'preferred_work_mode_onsite', 'preferred_work_mode_flexible'
        ]
    }

    # Save feature metadata early so test script can rely on it
    feats_path = os.path.join(BASE_DIR, 'models', 'feature_metadata.json')
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    with open(feats_path, 'w') as fh:
        json.dump(feature_metadata, fh)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = []
        features_all = []
        for line in tqdm(f):
            try:
                cand = json.loads(line)
            except Exception:
                continue
            lines.append(cand)

            if len(lines) >= chunk_size:
                # Prepare texts and embeddings
                jd_emb = encoder.encode(jd_text, convert_to_tensor=True)

                summaries = [str(c.get('profile', {}).get('summary', '')) for c in lines]
                careers = [" ".join([ch.get('description','') for ch in c.get('career_history',[])]) for c in lines]
                skills_texts = [" ".join([s.get('name','') for s in c.get('skills',[])]) for c in lines]

                sum_embs = encoder.encode(summaries, convert_to_tensor=True)
                career_embs = encoder.encode(careers, convert_to_tensor=True)
                skills_embs = encoder.encode(skills_texts, convert_to_tensor=True)

                sim_sum = util.cos_sim(jd_emb, sum_embs).cpu().numpy().flatten()
                sim_career = util.cos_sim(jd_emb, career_embs).cpu().numpy().flatten()
                sim_skills = util.cos_sim(jd_emb, skills_embs).cpu().numpy().flatten()
                ai_skill_counts = [count_ai_skills_semantic(c.get('skills', []), ai_anchor_embs) for c in lines]

                X = []
                y = []
                for c, ssum, scar, ssk, ai_count in zip(lines, sim_sum, sim_career, sim_skills, ai_skill_counts):
                    row = build_feature_row(c, ssum, scar, ssk, ai_count)
                    X.append({k: row[k] for k in feature_metadata['features']})
                    row_with_id = {'candidate_id': c.get('candidate_id')}
                    row_with_id.update(row)
                    row_with_id['recruiter_response_rate'] = float(c.get('redrob_signals', {}).get('recruiter_response_rate', 0))
                    features_all.append(row_with_id)
                    y.append(row_with_id['recruiter_response_rate'])

                Xdf = pd.DataFrame(X)
                dchunk = xgb.DMatrix(Xdf, label=y)
                params = {'objective': 'reg:squarederror', 'tree_method': 'hist'}
                if device == 'cuda':
                    params['device'] = 'gpu'
                    params['tree_method'] = 'hist'

                booster = xgb.train(params, dchunk, num_boost_round=100, xgb_model=booster)
                lines = []

        # Final partial chunk
        if len(lines) > 0:
            jd_emb = encoder.encode(jd_text, convert_to_tensor=True)
            summaries = [str(c.get('profile', {}).get('summary', '')) for c in lines]
            careers = [" ".join([ch.get('description','') for ch in c.get('career_history',[])]) for c in lines]
            skills_texts = [" ".join([s.get('name','') for s in c.get('skills',[])]) for c in lines]

            sum_embs = encoder.encode(summaries, convert_to_tensor=True)
            career_embs = encoder.encode(careers, convert_to_tensor=True)
            skills_embs = encoder.encode(skills_texts, convert_to_tensor=True)

            sim_sum = util.cos_sim(jd_emb, sum_embs).cpu().numpy().flatten()
            sim_career = util.cos_sim(jd_emb, career_embs).cpu().numpy().flatten()
            sim_skills = util.cos_sim(jd_emb, skills_embs).cpu().numpy().flatten()
            ai_skill_counts = [count_ai_skills_semantic(c.get('skills', []), ai_anchor_embs) for c in lines]

            X = []
            y = []
            for c, ssum, scar, ssk, ai_count in zip(lines, sim_sum, sim_career, sim_skills, ai_skill_counts):
                row = build_feature_row(c, ssum, scar, ssk, ai_count)
                X.append({k: row[k] for k in feature_metadata['features']})
                row_with_id = {'candidate_id': c.get('candidate_id')}
                row_with_id.update(row)
                row_with_id['recruiter_response_rate'] = float(c.get('redrob_signals', {}).get('recruiter_response_rate', 0))
                features_all.append(row_with_id)
                y.append(row_with_id['recruiter_response_rate'])

            Xdf = pd.DataFrame(X)
            dchunk = xgb.DMatrix(Xdf, label=y)
            params = {'objective': 'reg:squarederror', 'tree_method': 'hist'}
            if device == 'cuda':
                params['device'] = 'gpu'
                params['tree_method'] = 'hist'

            booster = xgb.train(params, dchunk, num_boost_round=100, xgb_model=booster)

    if booster:
        booster.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Feature metadata saved to {feats_path}")
        # Intentionally do not write per-candidate features to disk.
        # Features will be computed at runtime by `test_model.py` to avoid stale/mismatched data.
        print("Per-candidate features were NOT saved (runtime feature computation enabled).")

if __name__ == "__main__":
    train_model()
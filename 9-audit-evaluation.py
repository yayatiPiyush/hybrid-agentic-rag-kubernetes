"""Step 10: audit saved Hybrid RAG generations and local judge scores.

Standard library only. Never calls Ollama or regenerates answers.

python 9-audit-evaluation.py prepare
# fill human_review_12.csv, then:
python 9-audit-evaluation.py report
"""
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

DEFAULT_IDS = ['k8s_001', 'k8s_002', 'k8s_003', 'k8s_005',
               'k8s_011', 'k8s_024', 'k8s_025', 'k8s_026',
               'k8s_031', 'k8s_035', 'k8s_036', 'k8s_040']
FIELDS = ['id', 'question', 'human_faithfulness', 'human_answer_relevance',
          'answer_correct', 'citations_support_claims', 'unsupported_claims', 'notes']


def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def source_links(answer):
    """Recognize markdown hyperlinks to Kubernetes docs, not example command URLs.

    Intentionally conservative: plain-text references are not counted. This is
    a diagnostic heuristic, not a proof of citation support.
    """
    links = re.findall(r'\[[^\]]+\]\((https?://[^\s)]+)\)', answer)
    return sorted(set(url.rstrip('.,;') for url in links
                      if url.startswith('https://kubernetes.io/docs/')))


def prepare(args):
    generated = load_json(args.generations)
    judges = {r['id']: r for r in load_json(args.judges)}
    by_id = {r['id']: r for r in generated}
    missing = [i for i in DEFAULT_IDS if i not in by_id or i not in judges]
    if missing:
        raise ValueError(f'IDs missing from saved results: {missing}')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'human_review_12.csv'
    if csv_path.exists() and not args.overwrite:
        print(f'Preserving existing reviews: {csv_path}')
    else:
        with csv_path.open('w', encoding='utf-8-sig', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            for i in DEFAULT_IDS:
                wr.writerow({'id': i, 'question': by_id[i]['question']})
        print(f'Created editable review template: {csv_path}')

    p = out / 'human_review_guide.md'
    with p.open('w', encoding='utf-8') as fh:
        fh.write('# Step 10 — Human review of 12 RAG answers\n\n')
        fh.write('Use `human_review_12.csv` to enter independent assessments. '\
                 'Do **not** treat the judge scores below as ground truth.\n\n')
        fh.write('**Scoring:** `human_faithfulness`: 0–1 fraction of verifiable factual claims '\
                 'supported by the *shown retrieved excerpts*. `human_answer_relevance`: 0–1 '\
                 'how fully the answer addresses the question. `answer_correct` and '\
                 '`citations_support_claims`: 1=yes, 0=no, blank=uncertain. List unsupported '\
                 'claims and explain discrepancies in `notes`. A missing claim or truncated '\
                 'excerpt warrants a note, not automatic rejection.\n\n')
        fh.write('This sample is deliberately diagnostic (not randomly representative). '\
                 'The provided context contains **truncated retrieval snippets**; do not '\
                 'assert full-page support from excerpts alone.\n\n')
        for i in DEFAULT_IDS:
            rec, j = by_id[i], judges[i]
            docs = source_links(rec['answer'])
            known = set(rec.get('available_source_urls') or [])
            fh.write(f"## {i} — {rec['question']}\n\n")
            fh.write(f"**Local judge:** faithfulness {j.get('judge', {}).get('faithfulness', 'N/A')}, "
                     f"relevance {j.get('judge', {}).get('answer_relevance', 'N/A')}. "
                     f"**Source-document links:** {len(docs)}, "
                     f"unmatched source-document links: {len([v for v in docs if v not in known])}.\n\n")
            fh.write('### Answer\n\n' + rec['answer'] + '\n\n')
            fh.write('### Retrieved snippets (first 5,000 characters)\n\n```text\n'
                     + rec.get('retrieved_context', '')[:5000].replace('```', "'''" )
                     + '\n```\n\n')
            fh.write('---\n\n')
    print(f'Created full review pack: {p}')
    # Compare URL heuristics across all records; do not count localhost examples as citations.
    issues = []
    for r in generated:
        cited = source_links(r['answer'])
        unknown = sorted(set(cited) - set(r.get('available_source_urls') or []))
        if unknown or not cited:
            issues.append({'id': r['id'], 'source_document_links': cited,
                           'unknown_source_document_links': unknown,
                           'no_markdown_source_link': not cited})
    with (out / 'citation_diagnostics.json').open('w', encoding='utf-8') as fh:
        json.dump(issues, fh, indent=2)
    print(f'Created conservative citation diagnostics ({len(issues)} flagged records).')


def get_score(value):
    v = str(value or '').strip()
    if not v:
        return None
    x = float(v)
    if not 0 <= x <= 1:
        raise ValueError(f'Human score out of range [0,1]: {x}')
    return x


def report(args):
    judges = {r['id']: r for r in load_json(args.judges)}
    p = Path(args.output) / 'human_review_12.csv'
    with p.open(encoding='utf-8-sig', newline='') as fh:
        reviews = list(csv.DictReader(fh))
    rows = []
    for r in reviews:
        j = judges[r['id']]['judge']
        f = get_score(r.get('human_faithfulness'))
        a = get_score(r.get('human_answer_relevance'))
        rows.append({'id': r['id'], 'human_faithfulness': f,
                     'judge_faithfulness': j.get('faithfulness'),
                     'human_answer_relevance': a,
                     'judge_answer_relevance': j.get('answer_relevance'),
                     'faithfulness_abs_error': abs(f-j['faithfulness']) if f is not None else None,
                     'relevance_abs_error': abs(a-j['answer_relevance']) if a is not None else None,
                     'notes': r.get('notes', '')})
    evaluated = [r for r in rows if r['faithfulness_abs_error'] is not None]
    compared = [r for r in rows if r['relevance_abs_error'] is not None]
    summary = {'reviewed_faithfulness': len(evaluated),
               'reviewed_relevance': len(compared),
               'faithfulness_MAE_vs_human': mean(r['faithfulness_abs_error'] for r in evaluated) if evaluated else None,
               'relevance_MAE_vs_human': mean(r['relevance_abs_error'] for r in compared) if compared else None,
               'per_question': rows,
               'warning': 'Human reviews are a targeted diagnostic sample, not population accuracy. Scores are subjective.'}
    dest = Path(args.output) / 'human_judge_comparison.json'
    with dest.open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Reviewed: {len(evaluated)}/12 faithfulness; {len(compared)}/12 relevance')
    if evaluated:
        print('Mean absolute judge error (faithfulness):', round(summary['faithfulness_MAE_vs_human'], 3))
    if compared:
        print('Mean absolute judge error (relevance):', round(summary['relevance_MAE_vs_human'], 3))
    if not evaluated and not compared:
        print('No human scores yet. Fill human_review_12.csv and run report again.')
    print('Saved:', dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'report'])
    p.add_argument('--generations', default='generation_eval_results.json')
    p.add_argument('--judges', default='answer_quality_results.json')
    p.add_argument('--output', default='step10_audit')
    p.add_argument('--overwrite', action='store_true', help='Recreate blank CSV, destroying existing reviews')
    a = p.parse_args()
    if a.action == 'prepare': prepare(a)
    else: report(a)


if __name__ == '__main__':
    main()

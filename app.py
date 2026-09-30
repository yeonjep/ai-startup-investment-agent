import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from agents.common import configure_runtime, PROJECT_ROOT
from agents.config import MAX_CANDIDATES
from agents.graph import build_graph
from agents.services import LiveServices


def create_run_directory(root):
    folder = root / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='AI 반도체 스타트업 투자 평가')
    parser.add_argument('--domain', default='AI 반도체')
    parser.add_argument('--max-candidates', type=int, default=MAX_CANDIDATES)
    parser.add_argument('--candidates', type=Path, help='[{"name": "기업명"}] JSON; 없으면 웹 탐색')
    parser.add_argument('--output-dir', type=Path, help='결과 저장 루트. 하위에 매번 고유 실행 폴더 생성')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--scenario', choices=['invest', 'all_hold', 'missing', 'excluded', 'uncertain'], default='invest')
    args = parser.parse_args(argv)
    if not 1 <= args.max_candidates <= 10:
        parser.error('--max-candidates must be between 1 and 10')
    configure_runtime()
    if not args.demo and not all(os.getenv(k) for k in ('OPENAI_API_KEY', 'TAVILY_API_KEY')):
        parser.error('.env에 OPENAI_API_KEY, TAVILY_API_KEY를 설정하세요. API 없이 검증하려면 --demo.')
    candidates = json.loads(args.candidates.read_text(encoding='utf-8')) if args.candidates else []
    if not isinstance(candidates, list) or any(not isinstance(c, dict) or not isinstance(c.get('name'), str) or not c['name'].strip() for c in candidates):
        parser.error('후보 파일은 name 문자열을 포함하는 객체 배열이어야 합니다.')
    from agents.startup import deduplicate
    candidates = deduplicate(candidates)
    root = (args.output_dir or PROJECT_ROOT / 'outputs' / ('demo/' + args.scenario if args.demo else 'live')).resolve()
    output = create_run_directory(root)
    if args.demo:
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        os.environ['LANGSMITH_TRACING'] = 'false'
        from agents.demo import DemoServices
        services = DemoServices(args.scenario)
    else:
        services = LiveServices(output)
    initial = {'domain': args.domain, 'max_candidates': args.max_candidates, 'candidates': candidates,
               'output_dir': str(output), 'evaluated': [], 'warnings': [], 'diagnostics': [], 'demo': args.demo}
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'mode': 'demo' if args.demo else 'live', 'domain': args.domain, 'max_candidates': args.max_candidates,
                'llm_model': os.getenv('LLM_MODEL') or 'gpt-4.1-mini',
                'embedding_model': os.getenv('EMBEDDING_MODEL') or 'Qwen/Qwen3-Embedding-0.6B', 'nodes': []}
    write_json(output / 'run.json', manifest)
    print(f'실행 폴더: {output}', flush=True)
    result = initial
    try:
        for mode, event in build_graph(services).stream(initial, config={'recursion_limit': 200}, stream_mode=['values', 'updates']):
            if mode == 'values':
                result = event
                write_json(output / 'state.json', result)
            else:
                for node in event:
                    manifest['nodes'].append(node)
                    print(f'완료: {node}', flush=True)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'run.json', manifest)
    write_json(root / 'latest.json', {'run_dir': str(output), 'report_path': result['report_path']})
    print(f"평가 완료: {len(result.get('evaluated', []))}개 기업")
    for record in result.get('evaluated', []):
        print(f"{record['startup']}: {record['decision']}, {record['total_score']:.2f}점, 결측 {len(record['missing_evidence'])}/13")
    print(f"보고서: {result['report_path']}")
    return result


if __name__ == '__main__':
    main()

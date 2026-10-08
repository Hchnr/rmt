import runpy


def test_development_scoring_rejects_format_violations():
    module=runpy.run_path('scripts/evaluate_06b_development.py')
    score=module['score']
    assert score({'kind':'upper'}, 'FORESTS ARE GREEN.')
    assert not score({'kind':'upper'}, 'Forests are green.')
    assert not score({'kind':'upper'}, '森林')
    assert not score({'kind':'upper'}, '123')
    row={'kind':'json','value':'rain'}
    assert score(row, '{"topic":"rain"}')
    assert not score(row, '```json\n{"topic":"rain"}\n```')
    assert not score(row, '{"topic":"rain","extra":1}')
    row={'kind':'integer','value':17}
    assert score(row, '17')
    assert not score(row, 'The answer is 17.')
    correct=module['correct']
    assert correct(row, r'The answer is \boxed{17}.')
    assert not correct(row, r'17 is an intermediate value; answer: \boxed{18}.')

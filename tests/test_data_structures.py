from src.core.data_structures import Population


def test_population_generation():
    pop = Population(n_persons=10, age_min=0, age_max=100, random_seed=1)
    assert len(pop.persons) == 10
    assert all(0 <= p.age <= 100 for p in pop.persons)

# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def generate_model(solver_name, solver_options):

    module_mappings = {


        'artificialbeecolonyalgorithm-abc': ('niapy.algorithms.basic', 'abc', 'ArtificialBeeColonyAlgorithm'),
        'batalgorithm-ba': ('niapy.algorithms.basic', 'ba', 'BatAlgorithm'),
        'beesalgorithm-bea': ('niapy.algorithms.basic', 'bea', 'BeesAlgorithm'),
        'bacterialforagingoptimization-bfo': ('niapy.algorithms.basic', 'bfo', 'BacterialForagingOptimization'),
        'camel-ca': ('niapy.algorithms.basic', 'ca', 'Camel'),
        'camelalgorithm-ca': ('niapy.algorithms.basic', 'ca', 'CamelAlgorithm'),
        'clonalselectionalgorithm-clonalg': ('niapy.algorithms.basic', 'clonalg', 'ClonalSelectionAlgorithm'),
        'coralreefsoptimization-cro': ('niapy.algorithms.basic', 'cro', 'CoralReefsOptimization'),
        'cuckoosearch-cs': ('niapy.algorithms.basic', 'cs', 'CuckooSearch'),
        'catswarmoptimization-cso': ('niapy.algorithms.basic', 'cso', 'CatSwarmOptimization'),
        'differentialevolution-de': ('niapy.algorithms.basic', 'de', 'DifferentialEvolution'),
        'dynnpdifferentialevolution-de': ('niapy.algorithms.basic', 'de', 'DynNpDifferentialEvolution'),
        'agingnpdifferentialevolution-de': ('niapy.algorithms.basic', 'de', 'AgingNpDifferentialEvolution'),
        'multistrategydifferentialevolution-de': ('niapy.algorithms.basic', 'de', 'MultiStrategyDifferentialEvolution'),
        'dynnpmultistrategydifferentialevolution-de': ('niapy.algorithms.basic', 'de', 'DynNpMultiStrategyDifferentialEvolution'),
        'evolutionstrategy1p1-es': ('niapy.algorithms.basic', 'es', 'EvolutionStrategy1p1'),
        'evolutionstrategymp1-es': ('niapy.algorithms.basic', 'es', 'EvolutionStrategyMp1'),
        'evolutionstrategympl-es': ('niapy.algorithms.basic', 'es', 'EvolutionStrategyMpL'),
        'evolutionstrategyml-es': ('niapy.algorithms.basic', 'es', 'EvolutionStrategyML'),
        'fireflyalgorithm-fa': ('niapy.algorithms.basic', 'fa', 'FireflyAlgorithm'),
        'forestoptimizationalgorithm-foa': ('niapy.algorithms.basic', 'foa', 'ForestOptimizationAlgorithm'),
        'flowerpollinationalgorithm-fpa': ('niapy.algorithms.basic', 'fpa', 'FlowerPollinationAlgorithm'),
        'fish-fss': ('niapy.algorithms.basic', 'fss', 'Fish'),
        'fishschoolsearch-fss': ('niapy.algorithms.basic', 'fss', 'FishSchoolSearch'),
        'barebonesfireworksalgorithm-fwa': ('niapy.algorithms.basic', 'fwa', 'BareBonesFireworksAlgorithm'),
        'fireworksalgorithm-fwa': ('niapy.algorithms.basic', 'fwa', 'FireworksAlgorithm'),
        'enh-fwa': ('niapy.algorithms.basic', 'fwa', 'EnhancedFireworksAlgorithm'),
        'dynamicfireworksalgorithmgauss-fwa': ('niapy.algorithms.basic', 'fwa', 'DynamicFireworksAlgorithmGauss'),
        'dynamicfireworksalgorithm-fwa': ('niapy.algorithms.basic', 'fwa', 'DynamicFireworksAlgorithm'),
        'geneticalgorithm-ga': ('niapy.algorithms.basic', 'ga', 'GeneticAlgorithm'),
        'gravitationalsearchalgorithm-gsa': ('niapy.algorithms.basic', 'gsa', 'GravitationalSearchAlgorithm'),
        'glowwormswarmoptimization-gso': ('niapy.algorithms.basic', 'gso', 'GlowwormSwarmOptimization'),
        'glowwormswarmoptimizationv1-gso': ('niapy.algorithms.basic', 'gso', 'GlowwormSwarmOptimizationV1'),
        'glowwormswarmoptimizationv2-gso': ('niapy.algorithms.basic', 'gso', 'GlowwormSwarmOptimizationV2'),
        'glowwormswarmoptimizationv3-gso': ('niapy.algorithms.basic', 'gso', 'GlowwormSwarmOptimizationV3'),
        'greywolfoptimizer-gwo': ('niapy.algorithms.basic', 'gwo', 'GreyWolfOptimizer'),
        'harrishawksoptimization-hho': ('niapy.algorithms.basic', 'hho', 'HarrisHawksOptimization'),
        'harmonysearch-hs': ('niapy.algorithms.basic', 'hs', 'HarmonySearch'),
        'harmonysearchv1-hs': ('niapy.algorithms.basic', 'hs', 'HarmonySearchV1'),
        'krillherd-kh': ('niapy.algorithms.basic', 'kh', 'KrillHerd'),
        'lion-loa': ('niapy.algorithms.basic', 'loa', 'Lion'),
        'lionoptimizationalgorithm-loa': ('niapy.algorithms.basic', 'loa', 'LionOptimizationAlgorithm'),
        'monarchbutterflyoptimization-mbo': ('niapy.algorithms.basic', 'mbo', 'MonarchButterflyOptimization'),
        'mothflameoptimizer-mfo': ('niapy.algorithms.basic', 'mfo', 'MothFlameOptimizer'),
        'monkeykingevolutionv1-mke': ('niapy.algorithms.basic', 'mke', 'MonkeyKingEvolutionV1'),
        'monkeykingevolutionv2-mke': ('niapy.algorithms.basic', 'mke', 'MonkeyKingEvolutionV2'),
        'monkeykingevolutionv3-mke': ('niapy.algorithms.basic', 'mke', 'MonkeyKingEvolutionV3'),
        'particleswarmalgorithm-pso': ('niapy.algorithms.basic', 'pso', 'ParticleSwarmAlgorithm'),
        'particleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'ParticleSwarmOptimization'),
        'oppositionvelocityclampingparticleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'OppositionVelocityClampingParticleSwarmOptimization'),
        'centerparticleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'CenterParticleSwarmOptimization'),
        'mutatedparticleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'MutatedParticleSwarmOptimization'),
        'mutatedcenterparticleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'MutatedCenterParticleSwarmOptimization'),
        'mutatedcenterunifiedparticleswarmoptimization-pso': ('niapy.algorithms.basic', 'pso', 'MutatedCenterUnifiedParticleSwarmOptimization'),
        'comprehensivelearningparticleswarmoptimizer-pso': ('niapy.algorithms.basic', 'pso', 'ComprehensiveLearningParticleSwarmOptimizer'),
        'sinecosinealgorithm-sca': ('niapy.algorithms.basic', 'sca', 'SineCosineAlgorithm'),
        'hybridbatalgorithm-hba': ('niapy.algorithms.modified', 'hba', 'HybridBatAlgorithm'),
        'differentialevolutionmts-hde': ('niapy.algorithms.modified', 'hde', 'DifferentialEvolutionMTS'),
        'differentialevolutionmtsv1-hde': ('niapy.algorithms.modified', 'hde', 'DifferentialEvolutionMTSv1'),
        'dynnpdifferentialevolutionmts-hde': ('niapy.algorithms.modified', 'hde', 'DynNpDifferentialEvolutionMTS'),
        'dynnpdifferentialevolutionmtsv1-hde': ('niapy.algorithms.modified', 'hde', 'DynNpDifferentialEvolutionMTSv1'),
        'multistrategydifferentialevolutionmts-hde': ('niapy.algorithms.modified', 'hde', 'MultiStrategyDifferentialEvolutionMTS'),
        'multistrategydifferentialevolutionmtsv1-hde': ('niapy.algorithms.modified', 'hde', 'MultiStrategyDifferentialEvolutionMTSv1'),
        'dynnpmultistrategydifferentialevolutionmts-hde': ('niapy.algorithms.modified', 'hde', 'DynNpMultiStrategyDifferentialEvolutionMTS'),
        'dynnpmultistrategydifferentialevolutionmtsv1-hde': ('niapy.algorithms.modified', 'hde', 'DynNpMultiStrategyDifferentialEvolutionMTSv1'),
        'hybridselfadaptivebatalgorithm-hsaba': ('niapy.algorithms.modified', 'hsaba', 'HybridSelfAdaptiveBatAlgorithm'),
        'dev-ilshade': ('niapy.algorithms.modified', 'ilshade', 'ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution'),
        'selfadaptivedifferentialevolution-jde': ('niapy.algorithms.modified', 'jde', 'SelfAdaptiveDifferentialEvolution'),
        'multistrategyselfadaptivedifferentialevolution-jde': ('niapy.algorithms.modified', 'jde', 'MultiStrategySelfAdaptiveDifferentialEvolution'),
        'parameterfreebatalgorithm-plba': ('niapy.algorithms.modified', 'plba', 'ParameterFreeBatAlgorithm'),
        'adaptivebatalgorithm-saba': ('niapy.algorithms.modified', 'saba', 'AdaptiveBatAlgorithm'),
        'selfadaptivebatalgorithm-saba': ('niapy.algorithms.modified', 'saba', 'SelfAdaptiveBatAlgorithm'),
        'successhistoryadaptivedifferentialevolution-shade': ('niapy.algorithms.modified', 'shade', 'SuccessHistoryAdaptiveDifferentialEvolution'),
        'lpsrsuccesshistoryadaptivedifferentialevolution-shade': ('niapy.algorithms.modified', 'shade', 'LpsrSuccessHistoryAdaptiveDifferentialEvolution'),
        'anarchicsocietyoptimization-aso': ('niapy.algorithms.other', 'aso', 'AnarchicSocietyOptimization'),
        'hillclimbalgorithm-hc': ('niapy.algorithms.other', 'hc', 'HillClimbAlgorithm'),
        'multipletrajectorysearch-mts': ('niapy.algorithms.other', 'mts', 'MultipleTrajectorySearch'),
        'multipletrajectorysearchv1-mts': ('niapy.algorithms.other', 'mts', 'MultipleTrajectorySearchV1'),
        'neldermeadmethod-nmm': ('niapy.algorithms.other', 'nmm', 'NelderMeadMethod'),
        'randomsearch-rs': ('niapy.algorithms.other', 'rs', 'RandomSearch'),
        'simulatedannealing-sa': ('niapy.algorithms.other', 'sa', 'SimulatedAnnealing'),
    }

    module_name, class_name, model_name = module_mappings.get(solver_name, (None, None, None))

    if module_name and class_name and model_name:
    
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)
        model_object = getattr(model_class, model_name)(**solver_options)

    else:
    
        raise ValueError("Invalid solver name. Please refer to https://feloopy.readthedocs.io/en/latest/heuristic.html")

    return model_object
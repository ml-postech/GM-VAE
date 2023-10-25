def add_distribution_args(parser):
    group = parser.add_argument_group('EuclideanNormal')
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')

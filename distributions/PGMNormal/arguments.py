def add_distribution_args(parser):
    group = parser.add_argument_group('PGMNormal')
    group.add_argument('--c', type=float, default=-1.)
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')

def add_distribution_args(parser):
    group = parser.add_argument_group('LearnablePGMNormal')
    group.add_argument('--encoder_layer', type=str, choices=['Vanilla', 'Exp'], default='Vanilla')
    group.add_argument('--decoder_layer', type=str, choices=['Vanilla', 'Log'], default='Vanilla')

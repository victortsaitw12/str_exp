from .svtrencoder import SVTREncoder

def large_svtr(img_size=[48, 160], max_seq_len=40, out_channels=384):
    return SVTREncoder(
        img_size=img_size,
        max_seq_len=max_seq_len,
        out_channels=out_channels,
        embed_dims=[192, 256, 512],
        depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        mixer_types=['Local'] * 10 + ['Global'] * 11,
    )

def small_svtr(img_size=[32, 100], max_seq_len=25, out_channels=192):
    return SVTREncoder(
        img_size=img_size,
        max_seq_len=max_seq_len,
        out_channels=out_channels,
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=['Local'] * 8 + ['Global'] * 7,
    )

def tiny_svtr(img_size=[32, 100], max_seq_len=25, out_channels=192):
    return SVTREncoder(img_size=img_size,
        max_seq_len=max_seq_len,
        out_channels=out_channels)
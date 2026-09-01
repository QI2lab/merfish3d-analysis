from unittest.mock import Mock


def test_registration_exposes_configured_datastore_and_unset_tile() -> None:
    from merfish3danalysis.DataRegistration import DataRegistration

    datastore = Mock()
    datastore.tile_ids = ["tile0000"]
    datastore.round_ids = ["round001"]
    datastore.bit_ids = ["bit001"]
    datastore.channel_psfs = []

    registration = DataRegistration(datastore=datastore)

    assert registration.datastore is datastore
    assert registration.tile_id is None

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


def test_global_registration_config_contains_behavior_not_transform_keys() -> None:
    from merfish3danalysis.DataRegistration import GlobalRegistrationConfig

    config = GlobalRegistrationConfig()

    assert not hasattr(config, "transform_key")
    assert not hasattr(config, "new_transform_key")



def config_contains_required_fields(config,required_config_fields):
    for field in required_config_fields:
        if field in config:
            continue
        else:
            return False
            print("Error, ES missing required config")
            print("Required fields are: ",required_config_fields)
            raise "Require " + field

        return True



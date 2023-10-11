class SaveNameHolder:
    """
    Class to establish and obtain a name value. We have a setter to establish
    the name, and a getter to obtain it.

    Args:
        _save_name: str
    """
    _save_name = None

    @classmethod
    def set_save_name(cls, save_name):
        """
        Setter method. Given a save_name, this function assign the value to
        save_name, to share with all the classes.

        Args:
            save_name: str
        """
        cls._save_name = save_name

    @classmethod
    def get_save_name(cls):
        """
        Getter method for the '_save_name' atribute.

        Returns:
            '_save_name' current value
        """
        return cls._save_name

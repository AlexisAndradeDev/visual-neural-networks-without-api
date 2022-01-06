from flask import flash

def no_data(field_value):
    """
    Returns True if field_value has no data, else False.

    Args:
        field_value (str): Current value of the field as a string.

    Returns:
        bool: True if field_value has no data, else False.
    """    
    fail_msg = DataRequired().call(field_value)
    return True if fail_msg else False

def check_set_of_incompatible_validators(validators, incompatible_validator_classes):
    """
    Returns True if there are two or more validator classes that are 
    incompatible with each other; else False.

    Args:
        validators (list of app.playground.validators.Validator): List
            of validators instances.
        incompatible_validator_classes (set of app.playground.validators.Validator): 
            List of validator classes (not instances) that are incompatible 
            with each other.
    Returns:
        bool: True if there are two or more incompatible classes; else False.
    """    
    # store in a list booleans that indicate if an incompatible validator
    # class is in the validators list
    incompatible_validator_class_matches = [
        any(isinstance(validator, incompatible_validator_class) for validator in validators) \
            for incompatible_validator_class in incompatible_validator_classes
    ]

    # more than one incompatible class in the validators list
    if sum(incompatible_validator_class_matches) > 1:
        return True
    return False

def check_all_sets_of_incompatible_validators(validators):
    """
    Raises an AssertionError if there are validator classes that are 
    incompatible with each other.

    Args:
        validators (list of app.playground.validators.Validator): List
            of validators instances.
    """
    all_incompatible_validators_classes = [
        set([DataRequired, DataNotRequired, DataRequiredIfDataInOtherFieldInValues, 
        DataRequiredIfDataInOtherField]),
    ]

    for incompatible_validator_classes in all_incompatible_validators_classes:
        assert not check_set_of_incompatible_validators(
            validators=validators, 
            incompatible_validator_classes=incompatible_validator_classes,
        ), f"Found two or more incompatible validator classes. These incompatible classes might be: {incompatible_validator_classes}"

def validate_field(field_value, previous_value, field_name, validate_failed, validators):
    """
    Validates the value of the field. If a validation fails, it will
    show a flash message.

    Args:
        field_value (str): Current value of the field as a string.
        previous_value (Any): Previous value of the field.
        field_name (str): Name of the field.
        validate_failed (bool): Validation status of all the fields of a form. 
            True if any field validation has failed; else False.
        validators (list of app.playground.validators.Validator): List
            of validators instances.

    Returns:
        new_value (Any): If no validator failed, it will be the value of the
            field, else it will be previous_value.
        field_validate_failed (bool): If at least one validator failed, it will 
            be True; else, it will keep its original value (False or True).
    """
    field_failed = False

    check_all_sets_of_incompatible_validators(validators)

    messages = []
    for validator in validators:
        fail_msg = validator.call(field_value)

        if validator.skip_validation(field_value):
            field_failed = False
            messages = []
            break

        if fail_msg:
            field_failed = True
            msg = f"{fail_msg} Field: {field_name}"
            messages.append(msg)

            if isinstance(validator, (DataRequired, 
                        DataRequiredIfDataInOtherField, 
                        DataRequiredIfDataInOtherFieldInValues,
                    )):
                # delete all other messages
                messages = [msg]
                break

    for msg in messages:
        flash(msg, category="danger")

    if field_failed:
        validate_failed = True
    new_value = field_value if not field_failed else previous_value

    return new_value, validate_failed

class Validator():
    def call(self):
        msg = ""
        return msg
    def skip_validation(self, field_value):
        return False

class NoDashes(Validator):
    def call(self, field_value):
        msg = ""
        if "-" in str(field_value):
            msg = "Dashes are not allowed."
        return msg

class Number(Validator):
    def __init__(self, type_=""):
        """
        Args:
            type_ (str, optional): empty string, 'int' or 'float'. 
                Defaults to "".
        """        
        self.type_ = type_

    def call(self, field_value):
        """
        If the value of the field can not be converted into a number,
        a string fail message will be return.

        Args:
            field_value (str): Current value of the field as a string.
        """    
        try:
            float(field_value)
            msg = ""
        except ValueError as e:
            msg = "Must be a number."
        else:
            if self.type_ == "float" and not float(field_value) % 1:
                msg = "Must be float."
            elif self.type_ == "int" and float(field_value) % 1:
                msg = "Must be integer."
        return msg

class InList(Validator):
    def __init__(self, list_):
        self.list = list_

    def call(self, field_value):
        msg = ""
        if field_value not in self.list:
            msg = f"Selected value does not exist. Must be: {str(self.list)[1:-1]}"
        return msg

class DataRequired(Validator):
    def call(self, field_value):
        msg = ""
        if field_value is None or field_value in ["None", ""]:
            msg = f"Field must not be empty."
        return msg

class DataNotRequired(Validator):
    def skip_validation(self, field_value):
        return no_data(field_value)

class DataRequiredIfDataInOtherField(Validator):
    def __init__(self, other_field_value):
        self.other_field_value = other_field_value
    def call(self, field_value):
        msg = ""
        if no_data(field_value) and not no_data(self.other_field_value):
            msg = "Field must not be empty."
        return msg
    def skip_validation(self, field_value):
        if no_data(field_value) and no_data(self.other_field_value):
            return True
        return False

class DataRequiredIfDataInOtherFieldInValues(Validator):
    def __init__(self, other_field_value, possible_values):
        self.other_field_value = other_field_value
        self.possible_values = possible_values
    def call(self, field_value):
        """
        If this field is empty or None, and self.other_field_value is in
        the list of possible values, a string fail message will be return.

        Example: if you have an 'Optimizer' field, and the user selects
            RMSprop, a learning rate is required, so the user will
            have to fill the 'Learning rate' field. You will pass to
            validate_field() this validator with parameters
            other_field_value=request.form.get('optimizer')
            and possible_values=['RMSprop'], so that when you use the
            method call, it will check if the 'Learning rate' field
            is not empty. If it is empty and the value of the 'Optimizer'
            field is 'RMSprop', it will return a fail message, else it will 
            return an empty string.

        Args:
            field_value (str): Current value of the field as a string.
        Returns:
            A string that contains a fail message if the validator failed.

        """        
        msg = ""
        if no_data(field_value) and self.other_field_value in self.possible_values:
            msg = "Field must not be empty."
        return msg
    def skip_validation(self, field_value):
        """
        If the value of the other field is not in the list of possible values, 
        the validation will be skipped.

        Args:
            field_value (str): Current value of the field as a string.

        Returns:
            bool: True if the value of the other field is not in the list 
                of possible values; else, False.
        """        
        if self.other_field_value not in self.possible_values:
            return True
        return False

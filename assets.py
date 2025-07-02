from enum import StrEnum


class CancelTestDriveMessages(StrEnum):
    CONFIRM = "Do you confirm the cancel of test drive code {code}? [y/n]"
    NOT_CANCEL = "User gave up canceling, He want to do the test drive."
    ERROR_CANCEL = (
        "Error when canceling the test drive. Need to call do the dealership."
    )
    CANCELD = "Test drive canceled."

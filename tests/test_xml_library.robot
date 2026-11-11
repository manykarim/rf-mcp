*** Settings ***
Library    XML
Library    OperatingSystem
Test Setup    Copy Book Fixture

*** Variables ***
${BOOK_FILE}    ${CURDIR}${/}..${/}test_data${/}book.xml
${WORKING_BOOK_FILE}    ${OUTPUT DIR}${/}book-under-test.xml

*** Test Cases ***
Read Book Nodes And Attributes
    ${book_count}=    Get Element Count    ${WORKING_BOOK_FILE}    book
    Should Be Equal As Integers    ${book_count}    2
    Element Should Exist    ${WORKING_BOOK_FILE}    book[@id="bk-001"]
    Element Text Should Be    ${WORKING_BOOK_FILE}    Robot Framework Essentials    xpath=book[@id="bk-001"]/title
    Element Text Should Be    ${WORKING_BOOK_FILE}    Jane Doe    xpath=book[@id="bk-001"]/author
    Element Attribute Should Be    ${WORKING_BOOK_FILE}    year    2023    xpath=book[@id="bk-001"]
    Element Attribute Should Be    ${WORKING_BOOK_FILE}    available    true    xpath=book[@id="bk-001"]
    Element Attribute Should Be    ${WORKING_BOOK_FILE}    currency    USD    xpath=book[@id="bk-001"]/price

Add Book And Verify Persisted Node
    ${xml}=    Parse XML    ${WORKING_BOOK_FILE}
    Add Element    ${xml}    <book id="bk-003" year="2024" available="true"><title>XML Library in Practice</title><author>Alex Reader</author><genre>Testing</genre><price currency="USD">18.50</price></book>
    Save XML    ${xml}    ${WORKING_BOOK_FILE}
    ${reopened_xml}=    Parse XML    ${WORKING_BOOK_FILE}
    ${book_count}=    Get Element Count    ${reopened_xml}    book
    Should Be Equal As Integers    ${book_count}    3
    Element Should Exist    ${reopened_xml}    book[@id="bk-003"]
    Element Text Should Be    ${reopened_xml}    XML Library in Practice    xpath=book[@id="bk-003"]/title
    Element Attribute Should Be    ${reopened_xml}    year    2024    xpath=book[@id="bk-003"]

Update Book And Verify Persisted Changes
    ${xml}=    Parse XML    ${WORKING_BOOK_FILE}
    Set Element Text    ${xml}    Robot Framework XML Library    xpath=book[@id="bk-001"]/title
    Set Element Attribute    ${xml}    revision    2    xpath=book[@id="bk-001"]
    Save XML    ${xml}    ${WORKING_BOOK_FILE}
    ${reopened_xml}=    Parse XML    ${WORKING_BOOK_FILE}
    Element Text Should Be    ${reopened_xml}    Robot Framework XML Library    xpath=book[@id="bk-001"]/title
    Element Attribute Should Be    ${reopened_xml}    revision    2    xpath=book[@id="bk-001"]

*** Keywords ***
Copy Book Fixture
    Copy File    ${BOOK_FILE}    ${WORKING_BOOK_FILE}
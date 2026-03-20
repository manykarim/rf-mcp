*** Settings ***
Documentation    Test suite generated from session 5e9a29d6-a55d-4963-b240-7bb762fb8742 containing 4 test cases for automation.
Library         XML
Test Tags       automated    generated

*** Test Cases ***
Verify Library Structure
    ${xml} =    Parse Xml    tests/e2e/books_and_authors.xml
    ${lib_name} =    Get Element Attribute    ${xml}    name
    Should Be Equal    ${lib_name}    City Library
    ${year} =    Get Element Attribute    ${xml}    established
    Should Be Equal    ${year}    1995
    ${count} =    Get Element Count    ${xml}    authors/author
    ${book_count} =    Get Element Count    ${xml}    books/book
    Should Be Equal As Integers    ${count}    3
    Should Be Equal As Integers    ${book_count}    4

Verify Author Details
    ${author1} =    Get Element    ${xml}    authors/author[1]
    ${name} =    Get Element Text    ${author1}    name
    Should Be Equal    ${name}    J.K. Rowling
    ${nat} =    Get Element Attribute    ${author1}    nationality
    ${aid} =    Get Element Attribute    ${author1}    id
    Should Be Equal    ${nat}    British
    Should Be Equal    ${aid}    a1

Verify Book Details
    ${book1} =    Get Element    ${xml}    books/book[1]
    ${title} =    Get Element Text    ${book1}    title
    ${isbn} =    Get Element Attribute    ${book1}    isbn
    ${genre} =    Get Element Attribute    ${book1}    genre
    ${ref} =    Get Element Attribute    ${book1}    author_ref
    ${pages} =    Get Element Text    ${book1}    pages
    Should Be Equal    ${title}    Harry Potter and the Philosopher's Stone
    Should Be Equal    ${isbn}    978-0-7475-3269-9
    Should Be Equal    ${genre}    Fantasy
    Should Be Equal    ${ref}    a1
    Should Be Equal    ${pages}    223

Verify Book Availability
    ${books} =    Get Elements    ${xml}    books/book
    ${total} =    Get Length    ${books}
    Should Be Equal As Integers    ${total}    4
    ${book3} =    Get Element    ${xml}    books/book[3]
    ${avail} =    Get Element Text    ${book3}    available
    Should Be Equal    ${avail}    false

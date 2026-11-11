*** Settings ***
Documentation    End-to-end shopping and order-confirmation flow for Flowline Supply.
Resource         keywords/checkout.resource
Test Setup       Open Isolated Storefront
Test Teardown    Close Browser
Test Tags        e2e    storefront    checkout

*** Test Cases ***
Shopper Can Place An Order
    Browse Catalog And Add Product
    Verify Product Is In Cart
    Continue To Checkout
    Verify Checkout Summary
    Verify Required Checkout Fields
    Complete Checkout
    Verify Order Confirmation

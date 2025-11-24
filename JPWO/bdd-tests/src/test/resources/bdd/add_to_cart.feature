Feature: Add product to cart

  Background:

  Scenario: User adds a product to the cart
    Given I am logged in as "customer3@practicesoftwaretesting.com" with password "pass123"
    Given I am on the home page
    When I open the first product
    And I add the product to the cart
    Then the cart should contain this product

Feature: Favorite products

  Scenario: User adds first product to favorites
    Given I am logged in as "customer3@practicesoftwaretesting.com" with password "pass123"
    And I open the home page
    And I open the first product
    And I add the product to favorites
    Then the favorites page should contain that product

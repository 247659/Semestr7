Feature: Remove product from favorites

  Scenario: User removes first product from favorites
    Given I am logged in as "customer3@practicesoftwaretesting.com" with password "pass123"
    When I open the home page
    And I open the first product
    And I add the product to favorites
    And I remove the product from favorites
    Then the favorites page should not contain that product

Feature: Add new category

  Scenario: Admin adds a new category
    Given I am logged in as admin "admin@practicesoftwaretesting.com" with password "welcome01"
    When I open the add category page
    And I create a new category
    Then the category should be visible in the admin categories list

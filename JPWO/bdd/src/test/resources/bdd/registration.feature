Feature: User Registration

  Scenario: Successful user registration
    Given I am on the registration page
    When I fill the registration form with valid data
    And I submit the registration form
    And I log in with the newly created account
    Then I should see the account page
    And the account title should contain "My account"

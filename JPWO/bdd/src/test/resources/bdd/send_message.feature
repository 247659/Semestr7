Feature: Send a message from Contact Page
  As a logged in customer
  I want to send a message
  So that it appears in my messages list

  Scenario: Successfully sending a message
    Given I am logged in as "customer3@practicesoftwaretesting.com" with password "pass123"
    And I am on the contact page
    When I select subject "Payments"
    And I type a unique message
    And I send the message
    Then I should see the message in the messages list
    And the message status should be "NEW"
    And the message subject should be "payments"

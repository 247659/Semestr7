package bdd.steps;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SendMessageSteps {
    private String generatedMessage;
    private String uniqueUUID;

    @Given("I am on the contact page")
    public void iAmOnContactPage() {
        SharedClass.contactPage.open();
    }

    @When("I select subject {string}")
    public void iSelectSubject(String subject) {
        SharedClass.contactPage.selectSubject(subject);
    }

    @When("I type a unique message")
    public void iTypeUniqueMessage() {
        uniqueUUID = java.util.UUID.randomUUID().toString();
        generatedMessage = uniqueUUID + "text with 50 letters abcdefghijklmnoprstuvwxyzABCD";
        SharedClass.contactPage.enterMessage(generatedMessage);
    }

    @When("I send the message")
    public void iSendTheMessage() {
        SharedClass.contactPage.sendMessage();
    }

    @Then("I should see the message in the messages list")
    public void iShouldSeeTheMessageInMessagesList() {
        SharedClass.messagesPage.open();
        String messageText = SharedClass.messagesPage.getFirstMessage();
        assertTrue(messageText.contains(uniqueUUID), "Message content does not match");
    }

    @Then("the message status should be {string}")
    public void theMessageStatusShouldBe(String expectedStatus) {
        assertEquals(expectedStatus, SharedClass.messagesPage.getFirstStatus());
    }

    @Then("the message subject should be {string}")
    public void theMessageSubjectShouldBe(String expectedSubject) {
        assertTrue(
                SharedClass.messagesPage.getFirstSubject().toLowerCase().contains(expectedSubject.toLowerCase()),
                "Subject mismatch"
        );
    }
}

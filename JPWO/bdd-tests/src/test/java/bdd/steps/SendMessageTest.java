package bdd.steps;

import bdd.pages.contact.ContactPage;
import bdd.pages.contact.MessagesPage;
import bdd.utils.DriverFactory;
import bdd.pages.user.LoginPage;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;
import org.openqa.selenium.WebDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SendMessageTest {
    private WebDriver driver = DriverFactory.getDriver();
    private ContactPage contactPage;
    private MessagesPage messagesPage;

    private String generatedMessage;
    private String uniqueUUID;

    @Given("I am logged in as {string} with password {string} - 2")
    public void iAmLoggedIn(String email, String password) {
        driver.get("http://localhost:4200/auth/login");
        LoginPage loginPage = new LoginPage(driver);
        loginPage.loginAs(email, password);

        contactPage = new ContactPage(driver);
        messagesPage = new MessagesPage(driver);
    }

    @Given("I am on the contact page")
    public void iAmOnContactPage() {
        contactPage.open();
    }

    @When("I select subject {string}")
    public void iSelectSubject(String subject) {
        contactPage.selectSubject(subject);
    }

    @When("I type a unique message")
    public void iTypeUniqueMessage() {
        uniqueUUID = java.util.UUID.randomUUID().toString();
        generatedMessage = uniqueUUID + "text with 50 letters abcdefghijklmnoprstuvwxyzABCD";
        contactPage.enterMessage(generatedMessage);
    }

    @When("I send the message")
    public void iSendTheMessage() {
        contactPage.sendMessage();
    }

    @Then("I should see the message in the messages list")
    public void iShouldSeeTheMessageInMessagesList() {
        messagesPage.open();
        String messageText = messagesPage.getFirstMessage();
        assertTrue(messageText.contains(uniqueUUID), "Message content does not match");
    }

    @Then("the message status should be {string}")
    public void theMessageStatusShouldBe(String expectedStatus) {
        assertEquals(expectedStatus, messagesPage.getFirstStatus());
    }

    @Then("the message subject should be {string}")
    public void theMessageSubjectShouldBe(String expectedSubject) {
        assertTrue(
                messagesPage.getFirstSubject().toLowerCase().contains(expectedSubject.toLowerCase()),
                "Subject mismatch"
        );
    }
}

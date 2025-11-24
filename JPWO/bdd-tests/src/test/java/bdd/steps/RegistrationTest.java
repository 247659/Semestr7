package bdd.steps;

import bdd.pages.account.AccountPage;
import bdd.pages.user.LoginPage;
import bdd.pages.user.RegistrationPage;
import bdd.utils.DriverFactory;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;
import org.openqa.selenium.WebDriver;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class RegistrationTest {

    private WebDriver driver;
    private RegistrationPage registrationPage;
    private LoginPage loginPage;
    private AccountPage accountPage;

    private String generatedEmail;
    private final String PASSWORD = "Flowers123XDDD!";

    @Given("I am on the registration page")
    public void i_am_on_the_registration_page() {
        driver = DriverFactory.getDriver();
        registrationPage = new RegistrationPage(driver);
        registrationPage.open();

        loginPage = new LoginPage(driver);
        accountPage = new AccountPage(driver);
    }

    @When("I fill the registration form with valid data")
    public void i_fill_the_registration_form_with_valid_data() {
        generatedEmail = "user" + System.currentTimeMillis() + "@example.com";

        registrationPage.fillForm(
                "Name",
                "Surname",
                "2000-01-01",
                "Street",
                "00-100",
                "City",
                "State",
                "Argentina",
                "123456789",
                generatedEmail,
                PASSWORD
        );
    }

    @When("I submit the registration form")
    public void i_submit_the_registration_form() {
        registrationPage.submitForm();
    }

    @When("I log in with the newly created account")
    public void i_log_in_with_the_newly_created_account() {
        loginPage.loginAs(generatedEmail, PASSWORD);
    }

    @Then("I should see the account page")
    public void i_should_see_the_account_page() {
        assertTrue(accountPage.isLoaded(), "Account page did not load properly");
    }

    @Then("the account title should contain {string}")
    public void the_account_title_should_contain(String expected) {
        assertTrue(accountPage.getTitle().contains(expected),
                "Account title mismatch");
    }
}

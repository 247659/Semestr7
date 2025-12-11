package bdd.pages.user;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class RegistrationPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By submitButton = By.cssSelector("[data-test='register-submit']");

    private final By firstName = By.cssSelector("[data-test='first-name']");
    private final By lastName = By.cssSelector("[data-test='last-name']");
    private final By birthDate = By.cssSelector("[data-test='dob']");
    private final By street = By.cssSelector("[data-test='street']");
    private final By postalCode = By.cssSelector("[data-test='postal_code']");
    private final By city = By.cssSelector("[data-test='city']");
    private final By state = By.cssSelector("[data-test='state']");
    private final By country = By.cssSelector("[data-test='country']");
    private final By phone = By.cssSelector("[data-test='phone']");
    private final By email = By.cssSelector("[data-test='email']");
    private final By password = By.cssSelector("[data-test='password']");
    private final By dob = By.cssSelector("[data-test='dob']");

    public RegistrationPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/auth/register");
    }

    private void type(By locator, String value) {
        WebElement element = wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
        element.clear();
        element.sendKeys(value);
    }

    private void select(By locator, String visibleText) {
        WebElement element = wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
        new Select(element).selectByVisibleText(visibleText);
    }

    public void setFirstName(String value) { type(firstName, value); }
    public void setLastName(String value) { type(lastName, value); }
    public void setBirthDate(String value) { type(birthDate, value); }
    public void setStreet(String value) { type(street, value); }
    public void setPostalCode(String value) { type(postalCode, value); }
    public void setCity(String value) { type(city, value); }
    public void setState(String value) { type(state, value); }
    public void setCountry(String value) { select(country, value); }
    public void setPhone(String value) { type(phone, value); }
    public void setEmail(String value) { type(email, value); }
    public void setPassword(String value) { type(password, value); }

    public void fillForm(
            String firstName,
            String lastName,
            String birthDate,
            String street,
            String postalCode,
            String city,
            String state,
            String country,
            String phone,
            String email,
            String password
    ) {
        setFirstName(firstName);
        setLastName(lastName);
        setBirthDate(birthDate);
        setStreet(street);
        setPostalCode(postalCode);
        setCity(city);
        setState(state);
        setCountry(country);
        setPhone(phone);
        setEmail(email);
        setPassword(password);
    }

    public void submitForm() {
        wait.until(ExpectedConditions.elementToBeClickable(submitButton)).click();
    }

}
package com.ui.pages.contact;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

public class ContactPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By subjectSelect = By.cssSelector("[data-test='subject']");
    private final By messageField = By.cssSelector("[data-test='message']");
    private final By sendButton = By.cssSelector("[data-test='contact-submit']");
    private final By successMessage = By.className("alert-success");

    public ContactPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/contact");
    }

    public void selectSubject(String imageName) {
        WebElement selectElement = wait.until(ExpectedConditions.visibilityOfElementLocated(subjectSelect));
        Select select = new Select(selectElement);
        select.selectByVisibleText(imageName);
    }

    public void enterMessage(String message) {
        WebElement element = driver.findElement(messageField);
        element.clear();
        element.sendKeys(message);
    }

    public void sendMessage() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(sendButton));
        button.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(successMessage));
    }
}

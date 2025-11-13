package com.ui.pages.contact;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class MessagesPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By tableRows = By.cssSelector("table.table-hover");

    public MessagesPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/account/messages");
        waitUntilTableIsLoaded();    }

    private void waitUntilTableIsLoaded() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(tableRows));
        wait.until(driver -> !driver.findElements(tableRows).isEmpty());
    }

    public String getMessage(String partialMessage) {
        String xpath = String.format("//tr[td[contains(normalize-space(.), '%s')]]/td[2]", partialMessage.substring(0, 10));
        return driver.findElement(By.xpath(xpath)).getText();
    }

    public String getStatus(String partialMessage) {
        String xpath = String.format("//tr[td[contains(normalize-space(.), '%s')]]/td[3]", partialMessage.substring(0, 10));
        return driver.findElement(By.xpath(xpath)).getText();
    }

    public String getSubject(String partialMessage) {
        String xpath = String.format("//tr[td[contains(normalize-space(.), '%s')]]/td[1]", partialMessage.substring(0, 10));
        return driver.findElement(By.xpath(xpath)).getText().toLowerCase();
    }
}
